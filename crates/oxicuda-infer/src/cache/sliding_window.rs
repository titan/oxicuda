//! # Attention-Sink Sliding-Window KV Cache (StreamingLLM)
//!
//! Implements the rolling KV-cache eviction policy from **StreamingLLM** (Xiao
//! et al., 2023, *"Efficient Streaming Language Models with Attention Sinks"*).
//!
//! ## Problem
//!
//! A naïve sliding-window cache that simply drops the oldest tokens collapses
//! catastrophically: transformer attention dumps a large amount of probability
//! mass onto the very first tokens ("attention sinks"), and evicting them
//! destabilises the softmax. StreamingLLM fixes this by *always* retaining the
//! first `n_sink` tokens, and keeping only the most recent `window` tokens after
//! them — giving a constant-memory cache that supports effectively unbounded
//! generation with no perplexity blow-up.
//!
//! ## What this manages
//!
//! [`SlidingWindowManager`] tracks, per sequence, which **logical token
//! positions** are currently resident, and decides — as each new token is
//! appended — which (if any) older position must be evicted to honour the
//! `n_sink + window` budget. It maps that policy onto the paged
//! [`crate::cache::kv_cache`] block granularity: a physical block becomes
//! reclaimable once *every* logical position it stores has been evicted.
//!
//! The manager owns no KV bytes itself; it returns [`BlockId`]s for the caller
//! to `dec_ref` on the underlying [`crate::cache::kv_cache::PagedKvCache`].

use crate::cache::kv_cache::BlockId;
use crate::error::{InferError, InferResult};
use std::collections::VecDeque;

// ─── SlidingWindowConfig ─────────────────────────────────────────────────────

/// StreamingLLM cache budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlidingWindowConfig {
    /// Number of initial "attention sink" tokens to retain permanently.
    pub n_sink: usize,
    /// Number of most-recent tokens to keep after the sinks (the rolling window).
    pub window: usize,
    /// Tokens per physical KV block (must match the cache's `block_size`).
    pub block_size: usize,
}

impl SlidingWindowConfig {
    /// Construct a validated config.
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if `window == 0` or `block_size == 0`.
    pub fn new(n_sink: usize, window: usize, block_size: usize) -> InferResult<Self> {
        if window == 0 {
            return Err(InferError::InvalidConfig("sliding window must be >= 1"));
        }
        if block_size == 0 {
            return Err(InferError::InvalidConfig("block_size must be >= 1"));
        }
        Ok(Self {
            n_sink,
            window,
            block_size,
        })
    }

    /// Maximum number of logical positions resident at once.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.n_sink + self.window
    }
}

// ─── ResidentToken ───────────────────────────────────────────────────────────

/// Bookkeeping for one resident logical position.
#[derive(Debug, Clone, Copy)]
struct ResidentToken {
    /// Logical position in the full (un-evicted) sequence: 0, 1, 2, …
    position: usize,
    /// Physical block storing this token's K/V.
    block: BlockId,
}

// ─── SlidingWindowManager ────────────────────────────────────────────────────

/// Per-sequence rolling KV state under the StreamingLLM policy.
///
/// Maintains two FIFO structures:
/// * `sinks` — the first `n_sink` positions, never evicted.
/// * `window` — the most-recent positions, evicted oldest-first once full.
pub struct SlidingWindowManager {
    config: SlidingWindowConfig,
    /// Retained sink tokens (positions `0..n_sink`).
    sinks: Vec<ResidentToken>,
    /// Rolling window of recent tokens (FIFO, evict from the front).
    window: VecDeque<ResidentToken>,
    /// Total tokens ever appended (the next position to assign).
    next_position: usize,
    /// Reference count of *live* logical tokens per physical block. When a
    /// block's count drops to zero it is reclaimable.
    block_live: Vec<(BlockId, usize)>,
}

impl SlidingWindowManager {
    /// Create a manager for a fresh sequence.
    #[must_use]
    pub fn new(config: SlidingWindowConfig) -> Self {
        Self {
            config,
            sinks: Vec::with_capacity(config.n_sink),
            window: VecDeque::with_capacity(config.window),
            next_position: 0,
            block_live: Vec::new(),
        }
    }

    /// Number of logical tokens currently resident (sinks + window).
    #[must_use]
    pub fn n_resident(&self) -> usize {
        self.sinks.len() + self.window.len()
    }

    /// Total tokens appended so far (including evicted ones).
    #[must_use]
    pub fn total_seen(&self) -> usize {
        self.next_position
    }

    /// Logical positions currently resident, in attention order
    /// (sinks first, then the rolling window oldest-to-newest). This is the
    /// position list the attention kernel should gather K/V for.
    #[must_use]
    pub fn resident_positions(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.sinks.iter().map(|t| t.position).collect();
        v.extend(self.window.iter().map(|t| t.position));
        v
    }

    /// Append a new token stored in physical `block`.
    ///
    /// Returns the [`BlockId`]s that became fully evicted as a result (their
    /// every resident position is gone) so the caller can `dec_ref` them. The
    /// returned vec is usually empty and is non-empty only at block boundaries.
    pub fn append(&mut self, block: BlockId) -> Vec<BlockId> {
        let position = self.next_position;
        self.next_position += 1;
        self.add_block_ref(block);

        let tok = ResidentToken { position, block };

        // Fill the sink region first.
        if self.sinks.len() < self.config.n_sink {
            self.sinks.push(tok);
            return Vec::new();
        }

        // Otherwise the token enters the rolling window.
        self.window.push_back(tok);
        if self.window.len() <= self.config.window {
            return Vec::new();
        }
        // Window overflow → evict the oldest windowed token.
        let evicted = self
            .window
            .pop_front()
            .expect("window non-empty after overflow");
        self.release_block_ref(evicted.block).into_iter().collect()
    }

    /// Is logical `position` currently resident in the cache?
    #[must_use]
    pub fn is_resident(&self, position: usize) -> bool {
        self.sinks.iter().any(|t| t.position == position)
            || self.window.iter().any(|t| t.position == position)
    }

    /// All physical blocks still holding at least one live token.
    #[must_use]
    pub fn live_blocks(&self) -> Vec<BlockId> {
        self.block_live
            .iter()
            .filter(|(_, c)| *c > 0)
            .map(|(b, _)| *b)
            .collect()
    }

    // ── Internal refcounting ─────────────────────────────────────────────────

    fn add_block_ref(&mut self, block: BlockId) {
        if let Some(entry) = self.block_live.iter_mut().find(|(b, _)| *b == block) {
            entry.1 += 1;
        } else {
            self.block_live.push((block, 1));
        }
    }

    /// Decrement a block's live count; return it if it just hit zero.
    fn release_block_ref(&mut self, block: BlockId) -> Option<BlockId> {
        let entry = self.block_live.iter_mut().find(|(b, _)| *b == block)?;
        if entry.1 > 0 {
            entry.1 -= 1;
        }
        if entry.1 == 0 {
            // Remove the dead entry to keep `block_live` compact.
            self.block_live.retain(|(b, _)| *b != block);
            Some(block)
        } else {
            None
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_sink: usize, window: usize, bs: usize) -> SlidingWindowConfig {
        SlidingWindowConfig::new(n_sink, window, bs).expect("valid config")
    }

    #[test]
    fn config_validates() {
        assert!(SlidingWindowConfig::new(4, 0, 16).is_err());
        assert!(SlidingWindowConfig::new(4, 8, 0).is_err());
        assert_eq!(cfg(4, 8, 16).capacity(), 12);
    }

    #[test]
    fn sinks_filled_first_no_eviction() {
        let mut m = SlidingWindowManager::new(cfg(2, 4, 1));
        // First two appends fill sinks; no eviction.
        assert!(m.append(BlockId(0)).is_empty());
        assert!(m.append(BlockId(1)).is_empty());
        assert_eq!(m.n_resident(), 2);
        assert_eq!(m.resident_positions(), vec![0, 1]);
    }

    #[test]
    fn window_rolls_and_evicts_oldest_window_token() {
        // n_sink=1, window=2, block_size=1 (one token per block for clarity).
        let mut m = SlidingWindowManager::new(cfg(1, 2, 1));
        m.append(BlockId(0)); // pos 0 → sink
        m.append(BlockId(1)); // pos 1 → window
        m.append(BlockId(2)); // pos 2 → window (window full: [1,2])
        // capacity = 3; still resident.
        assert_eq!(m.n_resident(), 3);
        assert_eq!(m.resident_positions(), vec![0, 1, 2]);

        // pos 3 overflows the window → evict pos 1 (block 1).
        let evicted = m.append(BlockId(3));
        assert_eq!(evicted, vec![BlockId(1)], "oldest window token evicted");
        assert_eq!(m.resident_positions(), vec![0, 2, 3]);
        assert!(m.is_resident(0), "sink retained");
        assert!(!m.is_resident(1), "evicted position gone");
    }

    #[test]
    fn sink_never_evicted_under_long_stream() {
        let mut m = SlidingWindowManager::new(cfg(2, 3, 1));
        for i in 0..100 {
            m.append(BlockId(i as u32));
        }
        // Positions 0 and 1 are sinks → always resident.
        assert!(m.is_resident(0));
        assert!(m.is_resident(1));
        // Resident count is bounded by capacity.
        assert_eq!(m.n_resident(), 5);
        // Window holds the last 3 positions: 97, 98, 99.
        let pos = m.resident_positions();
        assert_eq!(pos, vec![0, 1, 97, 98, 99]);
        assert_eq!(m.total_seen(), 100);
    }

    #[test]
    fn block_freed_only_when_all_its_tokens_evicted() {
        // block_size=2: each physical block holds two logical tokens.
        // n_sink=0 so everything is windowed; window=2 tokens = 1 block worth.
        let mut m = SlidingWindowManager::new(cfg(0, 2, 2));
        // Two tokens share block 0.
        assert!(m.append(BlockId(0)).is_empty()); // pos0 → block0 (count 1)
        assert!(m.append(BlockId(0)).is_empty()); // pos1 → block0 (count 2)
        // window now full with [pos0,pos1].
        // pos2 in block 1 → evict pos0; block0 still has pos1 live → NOT freed.
        let e1 = m.append(BlockId(1));
        assert!(e1.is_empty(), "block0 still holds pos1");
        // pos3 in block 1 → evict pos1; block0 now fully dead → freed.
        let e2 = m.append(BlockId(1));
        assert_eq!(e2, vec![BlockId(0)], "block0 reclaimable once empty");
    }

    #[test]
    fn live_blocks_reflects_residency() {
        let mut m = SlidingWindowManager::new(cfg(1, 1, 1));
        m.append(BlockId(0)); // sink
        m.append(BlockId(1)); // window
        m.append(BlockId(2)); // evict block1, window now [block2]
        let mut live = m.live_blocks();
        live.sort();
        assert_eq!(live, vec![BlockId(0), BlockId(2)]);
    }

    #[test]
    fn zero_sink_pure_sliding_window() {
        let mut m = SlidingWindowManager::new(cfg(0, 2, 1));
        m.append(BlockId(0));
        m.append(BlockId(1));
        let e = m.append(BlockId(2)); // evict pos0
        assert_eq!(e, vec![BlockId(0)]);
        assert_eq!(m.resident_positions(), vec![1, 2]);
    }
}

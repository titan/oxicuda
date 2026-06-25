//! # Chunked Prefill (Sarathi)
//!
//! Implements the prompt-chunking schedule from **Sarathi** (Agrawal et al.,
//! 2023, *"Sarathi: Efficient LLM Inference by Piggybacking Decodes with
//! Chunked Prefills"*), also adopted by vLLM as `enable_chunked_prefill`.
//!
//! ## Why
//!
//! A long prompt processed in a single prefill forward pass monopolises the GPU
//! for one large step, stalling every concurrently-decoding sequence and
//! spiking inter-token latency (a "prefill bubble"). Sarathi splits the prompt
//! into fixed token-budget **chunks**; each scheduler step processes one prompt
//! chunk *together with* the decode tokens of other running sequences, so the
//! batch always fills the same compute budget and decode latency stays smooth.
//!
//! ## What this provides
//!
//! [`ChunkedPrefillPlan`] divides a prompt of `prompt_len` tokens into chunks of
//! at most `chunk_size` tokens, tracking progress as chunks are consumed.
//! [`ChunkPlanner::pack_step`] performs the core Sarathi packing decision: given
//! a per-step token budget and the set of decode sequences that must run, it
//! decides how many prefill tokens can "piggyback" in the remaining budget.

use crate::error::{InferError, InferResult};

// ─── PrefillChunk ────────────────────────────────────────────────────────────

/// One contiguous slice of a prompt to process in a single step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillChunk {
    /// Inclusive start position within the prompt.
    pub start: usize,
    /// Exclusive end position within the prompt.
    pub end: usize,
}

impl PrefillChunk {
    /// Number of tokens in this chunk.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Is the chunk empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

// ─── ChunkedPrefillPlan ──────────────────────────────────────────────────────

/// Stateful chunk cursor over a single prompt.
///
/// Yields successive [`PrefillChunk`]s of at most `chunk_size` tokens until the
/// whole prompt has been prefilled. A *variable* per-step amount may be taken
/// (via [`ChunkedPrefillPlan::take`]) to fill exactly the budget a step leaves
/// after its decodes — the essence of Sarathi piggybacking.
#[derive(Debug, Clone)]
pub struct ChunkedPrefillPlan {
    prompt_len: usize,
    chunk_size: usize,
    /// Tokens prefilled so far.
    cursor: usize,
}

impl ChunkedPrefillPlan {
    /// Create a plan for a `prompt_len`-token prompt with a maximum chunk of
    /// `chunk_size` tokens.
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if `chunk_size == 0`.
    pub fn new(prompt_len: usize, chunk_size: usize) -> InferResult<Self> {
        if chunk_size == 0 {
            return Err(InferError::InvalidConfig("chunk_size must be >= 1"));
        }
        Ok(Self {
            prompt_len,
            chunk_size,
            cursor: 0,
        })
    }

    /// Has the whole prompt been prefilled?
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.cursor >= self.prompt_len
    }

    /// Tokens still awaiting prefill.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.prompt_len.saturating_sub(self.cursor)
    }

    /// Tokens prefilled so far.
    #[must_use]
    pub fn progress(&self) -> usize {
        self.cursor
    }

    /// Total number of chunks if consumed at full `chunk_size`.
    #[must_use]
    pub fn n_chunks(&self) -> usize {
        self.prompt_len.div_ceil(self.chunk_size)
    }

    /// Take the next chunk of up to `chunk_size` tokens, advancing the cursor.
    /// Returns `None` once the prompt is fully prefilled.
    pub fn next_chunk(&mut self) -> Option<PrefillChunk> {
        self.take(self.chunk_size)
    }

    /// Take up to `max_tokens` (but never more than `chunk_size`, nor more than
    /// remain) prefill tokens, advancing the cursor. Returns `None` if no
    /// tokens are taken (budget zero or prompt finished).
    ///
    /// This is the variable-size path the Sarathi scheduler uses to *exactly*
    /// fill the budget a step leaves after its decodes.
    pub fn take(&mut self, max_tokens: usize) -> Option<PrefillChunk> {
        if self.is_done() {
            return None;
        }
        let take = max_tokens.min(self.chunk_size).min(self.remaining());
        if take == 0 {
            return None;
        }
        let chunk = PrefillChunk {
            start: self.cursor,
            end: self.cursor + take,
        };
        self.cursor += take;
        Some(chunk)
    }
}

// ─── ChunkPlanner ────────────────────────────────────────────────────────────

/// One step's packing decision: how prompt-chunk tokens piggyback on decodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepPacking {
    /// Number of decode sequences scheduled (one token each).
    pub n_decode: usize,
    /// Prefill chunk to process this step, if any budget remained.
    pub prefill_chunk: Option<PrefillChunk>,
    /// Total tokens consumed by the step (`n_decode + prefill chunk length`).
    pub total_tokens: usize,
}

/// Stateless helper implementing the Sarathi per-step packing rule.
#[derive(Debug, Clone, Copy)]
pub struct ChunkPlanner {
    /// Token budget per forward-pass step.
    pub max_batch_tokens: usize,
}

impl ChunkPlanner {
    /// Create a planner with the given per-step token budget.
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if `max_batch_tokens == 0`.
    pub fn new(max_batch_tokens: usize) -> InferResult<Self> {
        if max_batch_tokens == 0 {
            return Err(InferError::InvalidConfig("max_batch_tokens must be >= 1"));
        }
        Ok(Self { max_batch_tokens })
    }

    /// Pack one step: schedule `n_decode` decode tokens first (they are
    /// latency-critical and cheap), then fill the *remaining* budget with the
    /// next prompt chunk from `prefill`.
    ///
    /// Decodes always take priority; if they alone exceed the budget the prefill
    /// chunk is skipped this step (its tokens roll to a later step).
    pub fn pack_step(&self, n_decode: usize, prefill: &mut ChunkedPrefillPlan) -> StepPacking {
        let decode_cost = n_decode.min(self.max_batch_tokens);
        let budget_left = self.max_batch_tokens.saturating_sub(decode_cost);
        let prefill_chunk = if budget_left > 0 {
            prefill.take(budget_left)
        } else {
            None
        };
        let chunk_len = prefill_chunk.map_or(0, |c| c.len());
        StepPacking {
            n_decode,
            prefill_chunk,
            total_tokens: decode_cost + chunk_len,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_size_zero_rejected() {
        assert!(ChunkedPrefillPlan::new(10, 0).is_err());
    }

    #[test]
    fn full_chunks_cover_prompt() {
        // 10-token prompt, chunk_size 4 → chunks [0,4),[4,8),[8,10).
        let mut p = ChunkedPrefillPlan::new(10, 4).expect("ok");
        assert_eq!(p.n_chunks(), 3);
        assert_eq!(p.next_chunk(), Some(PrefillChunk { start: 0, end: 4 }));
        assert_eq!(p.next_chunk(), Some(PrefillChunk { start: 4, end: 8 }));
        assert_eq!(p.next_chunk(), Some(PrefillChunk { start: 8, end: 10 }));
        assert_eq!(p.next_chunk(), None);
        assert!(p.is_done());
    }

    #[test]
    fn progress_and_remaining_track() {
        let mut p = ChunkedPrefillPlan::new(10, 4).expect("ok");
        assert_eq!(p.remaining(), 10);
        p.next_chunk();
        assert_eq!(p.progress(), 4);
        assert_eq!(p.remaining(), 6);
    }

    #[test]
    fn take_respects_budget_and_chunk_cap() {
        // chunk_size=4 caps even a larger budget request.
        let mut p = ChunkedPrefillPlan::new(20, 4).expect("ok");
        let c = p.take(100).expect("tokens available");
        assert_eq!(c.len(), 4, "never exceed chunk_size");
        // Small budget takes fewer.
        let c2 = p.take(2).expect("tokens available");
        assert_eq!(c2.len(), 2);
        assert_eq!(p.progress(), 6);
    }

    #[test]
    fn take_zero_budget_yields_none() {
        let mut p = ChunkedPrefillPlan::new(8, 4).expect("ok");
        assert!(p.take(0).is_none());
        assert_eq!(p.progress(), 0);
    }

    #[test]
    fn empty_prompt_done_immediately() {
        let mut p = ChunkedPrefillPlan::new(0, 4).expect("ok");
        assert!(p.is_done());
        assert!(p.next_chunk().is_none());
        assert_eq!(p.n_chunks(), 0);
    }

    #[test]
    fn planner_decode_priority_then_prefill() {
        let planner = ChunkPlanner::new(8).expect("ok");
        let mut prefill = ChunkedPrefillPlan::new(20, 16).expect("ok");
        // 3 decodes; budget 8 → 5 prefill tokens piggyback.
        let step = planner.pack_step(3, &mut prefill);
        assert_eq!(step.n_decode, 3);
        assert_eq!(step.prefill_chunk, Some(PrefillChunk { start: 0, end: 5 }));
        assert_eq!(step.total_tokens, 8, "budget fully packed");
    }

    #[test]
    fn planner_decodes_exhaust_budget_skip_prefill() {
        let planner = ChunkPlanner::new(4).expect("ok");
        let mut prefill = ChunkedPrefillPlan::new(20, 16).expect("ok");
        // 4 decodes saturate the budget → no prefill this step.
        let step = planner.pack_step(4, &mut prefill);
        assert_eq!(step.prefill_chunk, None);
        assert_eq!(step.total_tokens, 4);
        assert_eq!(prefill.progress(), 0, "prompt untouched");
    }

    #[test]
    fn planner_no_decodes_full_prefill_chunk() {
        let planner = ChunkPlanner::new(8).expect("ok");
        let mut prefill = ChunkedPrefillPlan::new(20, 16).expect("ok");
        // No decodes → whole budget goes to prefill (capped at chunk_size=16,
        // but budget is 8 → 8 tokens).
        let step = planner.pack_step(0, &mut prefill);
        assert_eq!(step.prefill_chunk, Some(PrefillChunk { start: 0, end: 8 }));
        assert_eq!(step.total_tokens, 8);
    }

    #[test]
    fn multi_step_drains_prompt_with_steady_budget() {
        let planner = ChunkPlanner::new(8).expect("ok");
        let mut prefill = ChunkedPrefillPlan::new(20, 16).expect("ok");
        // Each step has 2 decodes → 6 prefill tokens per step.
        // 20 prompt tokens / 6 ≈ 4 steps (6+6+6+2).
        let mut steps = 0;
        let mut prefilled = 0;
        while !prefill.is_done() {
            let step = planner.pack_step(2, &mut prefill);
            prefilled += step.prefill_chunk.map_or(0, |c| c.len());
            assert!(step.total_tokens <= 8, "never exceed budget");
            steps += 1;
            assert!(steps < 100, "must terminate");
        }
        assert_eq!(prefilled, 20, "entire prompt prefilled");
        assert_eq!(steps, 4);
    }

    #[test]
    fn chunk_len_and_empty() {
        let c = PrefillChunk { start: 2, end: 5 };
        assert_eq!(c.len(), 3);
        assert!(!c.is_empty());
        let e = PrefillChunk { start: 4, end: 4 };
        assert!(e.is_empty());
    }
}

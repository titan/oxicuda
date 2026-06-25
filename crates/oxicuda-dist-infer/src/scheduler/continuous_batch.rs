//! Continuous (iteration-level) batching scheduler.
//!
//! Classic static batching pads every request to the longest sequence and runs
//! the whole batch to completion, wasting compute on finished sequences.
//! **Continuous batching** (Orca, vLLM) instead schedules *per decode
//! iteration*: each step it (a) admits as many waiting requests as the paged-KV
//! block budget allows, runs their prefill; (b) advances every running sequence
//! by exactly one generated token; (c) retires sequences that hit their token
//! limit, freeing their blocks immediately for the next admission.
//!
//! This module models that loop over an in-memory block budget. It is the
//! scheduling *policy* only — no attention math — so it has an exact oracle:
//! the sum of all sequences' allocated blocks never exceeds capacity, FCFS +
//! priority admission order is respected, and each admitted sequence advances by
//! exactly the number of tokens it was generated for.

use std::collections::VecDeque;

use crate::error::{DistInferError, DistInferResult};
use crate::router::request::Request;

/// Lifecycle state of a sequence inside the batcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqState {
    /// In the waiting queue, prefill not yet run.
    Waiting,
    /// Prefill done, generating tokens.
    Running,
    /// Reached `max_new_tokens`; blocks freed.
    Finished,
}

/// Internal per-sequence accounting record.
#[derive(Debug, Clone)]
struct SeqRecord {
    request_id: u64,
    priority: i32,
    /// Number of prompt tokens (prefill length).
    prompt_len: usize,
    /// Tokens generated so far.
    generated: usize,
    /// Total tokens to generate before finishing.
    max_new_tokens: usize,
    /// KV blocks currently allocated to this sequence.
    blocks: usize,
    state: SeqState,
}

impl SeqRecord {
    /// Total tokens currently stored = prompt + generated.
    fn total_tokens(&self) -> usize {
        self.prompt_len + self.generated
    }

    /// Blocks required to hold `total_tokens` at `block_size` tokens/block.
    fn required_blocks(&self, block_size: usize) -> usize {
        self.total_tokens().div_ceil(block_size)
    }
}

/// One iteration's scheduling decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchPlan {
    /// Request ids admitted (prefilled) this iteration.
    pub admitted: Vec<u64>,
    /// Request ids advanced by one decode token this iteration.
    pub decoded: Vec<u64>,
    /// Request ids that finished (hit token limit) this iteration.
    pub finished: Vec<u64>,
    /// Free blocks remaining after this iteration.
    pub free_blocks: usize,
}

/// Iteration-level batching scheduler over a paged-KV block budget.
#[derive(Debug)]
pub struct ContinuousBatcher {
    /// Tokens stored per KV block.
    block_size: usize,
    /// Total KV blocks available.
    total_blocks: usize,
    /// Blocks currently allocated across all running sequences.
    used_blocks: usize,
    /// Maximum sequences allowed to run concurrently (batch width cap).
    max_batch_size: usize,
    /// Waiting queue (admission candidates), priority-then-FCFS ordered.
    waiting: VecDeque<SeqRecord>,
    /// Running sequences (generating).
    running: Vec<SeqRecord>,
}

impl ContinuousBatcher {
    /// Construct a batcher.
    ///
    /// # Errors
    ///
    /// * [`DistInferError::DimensionMismatch`] if `block_size == 0`,
    ///   `total_blocks == 0`, or `max_batch_size == 0`.
    pub fn new(
        block_size: usize,
        total_blocks: usize,
        max_batch_size: usize,
    ) -> DistInferResult<Self> {
        if block_size == 0 || total_blocks == 0 || max_batch_size == 0 {
            return Err(DistInferError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        Ok(Self {
            block_size,
            total_blocks,
            used_blocks: 0,
            max_batch_size,
            waiting: VecDeque::new(),
            running: Vec::new(),
        })
    }

    /// Enqueue a request. It is inserted by descending priority; ties keep FCFS
    /// (stable — inserted after equal-priority requests already queued).
    pub fn submit(&mut self, req: &Request) {
        let rec = SeqRecord {
            request_id: req.request_id,
            priority: req.priority,
            prompt_len: req.token_ids.len().max(1),
            generated: 0,
            max_new_tokens: req.max_new_tokens,
            blocks: 0,
            state: SeqState::Waiting,
        };
        // Find insertion point: after all strictly-higher-priority records.
        let pos = self
            .waiting
            .iter()
            .position(|r| r.priority < rec.priority)
            .unwrap_or(self.waiting.len());
        self.waiting.insert(pos, rec);
    }

    /// Free blocks available right now.
    #[must_use]
    pub fn free_blocks(&self) -> usize {
        self.total_blocks - self.used_blocks
    }

    /// Number of currently running sequences.
    #[must_use]
    pub fn n_running(&self) -> usize {
        self.running.len()
    }

    /// Number of waiting sequences.
    #[must_use]
    pub fn n_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Look up a sequence's state (Waiting / Running / Finished / unknown).
    #[must_use]
    pub fn state_of(&self, request_id: u64) -> Option<SeqState> {
        if let Some(r) = self.running.iter().find(|r| r.request_id == request_id) {
            return Some(r.state);
        }
        if self.waiting.iter().any(|r| r.request_id == request_id) {
            return Some(SeqState::Waiting);
        }
        None
    }

    /// Run one scheduling iteration.
    ///
    /// Order of operations (the Orca/vLLM policy):
    /// 1. **Decode** every running sequence by one token; this may push a
    ///    sequence onto a new block (allocated if budget allows, else the
    ///    sequence is preempted back to *waiting* and its blocks freed).
    /// 2. **Retire** sequences that reached `max_new_tokens`, freeing blocks.
    /// 3. **Admit** waiting sequences (highest priority first) whose prefill
    ///    blocks fit the remaining budget and the batch-width cap.
    ///
    /// Returns the [`BatchPlan`] describing what happened.
    pub fn step(&mut self) -> BatchPlan {
        let mut decoded = Vec::new();
        let mut finished = Vec::new();
        let mut preempted: Vec<SeqRecord> = Vec::new();

        // ── 1. Decode running sequences ───────────────────────────────────────
        let mut survivors: Vec<SeqRecord> = Vec::with_capacity(self.running.len());
        // Drain running so we can re-decide membership.
        let running = std::mem::take(&mut self.running);
        for mut rec in running {
            rec.generated += 1;
            let need = rec.required_blocks(self.block_size);
            if need > rec.blocks {
                // Needs another block.
                let delta = need - rec.blocks;
                if self.used_blocks + delta <= self.total_blocks {
                    self.used_blocks += delta;
                    rec.blocks = need;
                } else {
                    // Out of memory → preempt: roll back the token, free blocks,
                    // requeue as waiting (recomputed on re-admission).
                    rec.generated -= 1;
                    self.used_blocks -= rec.blocks;
                    rec.blocks = 0;
                    rec.state = SeqState::Waiting;
                    preempted.push(rec);
                    continue;
                }
            }
            decoded.push(rec.request_id);
            // ── 2. Retire if done ─────────────────────────────────────────────
            if rec.generated >= rec.max_new_tokens {
                self.used_blocks -= rec.blocks;
                rec.blocks = 0;
                rec.state = SeqState::Finished;
                finished.push(rec.request_id);
            } else {
                survivors.push(rec);
            }
        }
        self.running = survivors;

        // Re-queue preempted sequences at the FRONT (they keep their priority
        // slot — preemption is for memory, not demotion). Their ids are excluded
        // from *this* iteration's admission so a sequence cannot bounce
        // running→preempt→running within one step; it retries next step once
        // other sequences may have freed memory.
        let preempted_ids: Vec<u64> = preempted.iter().map(|r| r.request_id).collect();
        for rec in preempted.into_iter().rev() {
            let pos = self
                .waiting
                .iter()
                .position(|r| r.priority < rec.priority)
                .unwrap_or(self.waiting.len());
            self.waiting.insert(pos, rec);
        }

        // ── 3. Admit waiting sequences ────────────────────────────────────────
        let mut admitted = Vec::new();
        // Skip any sequence preempted this iteration; iterate from the front,
        // honouring head-of-line blocking for non-preempted candidates.
        let mut scan = 0usize;
        while self.running.len() < self.max_batch_size && scan < self.waiting.len() {
            let cand = &self.waiting[scan];
            if preempted_ids.contains(&cand.request_id) {
                scan += 1;
                continue;
            }
            let need = cand.required_blocks(self.block_size);
            if self.used_blocks + need > self.total_blocks {
                break; // head-of-line blocked; preserve FCFS within priority.
            }
            // Commit admission: remove from the waiting queue at `scan`.
            let mut rec = self
                .waiting
                .remove(scan)
                .expect("index in range; remove must succeed");
            rec.generated = 0;
            rec.blocks = need;
            rec.state = SeqState::Running;
            self.used_blocks += need;
            admitted.push(rec.request_id);
            self.running.push(rec);
            // `scan` now points at the next candidate (elements shifted left).
        }

        BatchPlan {
            admitted,
            decoded,
            finished,
            free_blocks: self.free_blocks(),
        }
    }

    /// Run iterations until every submitted request has finished or no progress
    /// is possible, returning the number of iterations taken.
    ///
    /// Used in tests / offline simulation. Guards against livelock by capping at
    /// a generous bound derived from the total work.
    pub fn run_to_completion(&mut self) -> DistInferResult<usize> {
        let mut iters = 0usize;
        let cap = 1 + self
            .waiting
            .iter()
            .chain(self.running.iter())
            .map(|r| r.max_new_tokens + 1)
            .sum::<usize>()
            .max(1)
            * 4;
        loop {
            if self.waiting.is_empty() && self.running.is_empty() {
                return Ok(iters);
            }
            let before_running = self.running.len();
            let before_waiting = self.waiting.len();
            let plan = self.step();
            iters += 1;
            // Progress check: something must have happened.
            let progressed = !plan.admitted.is_empty()
                || !plan.decoded.is_empty()
                || !plan.finished.is_empty()
                || self.running.len() != before_running
                || self.waiting.len() != before_waiting;
            if !progressed {
                return Err(DistInferError::Internal(
                    "continuous batcher made no progress (deadlock: a request needs more blocks than exist)",
                ));
            }
            if iters > cap {
                return Err(DistInferError::Internal(
                    "continuous batcher exceeded iteration cap",
                ));
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn req(id: u64, prompt: usize, max_new: usize, priority: i32) -> Request {
        Request {
            request_id: id,
            token_ids: vec![1u32; prompt],
            max_new_tokens: max_new,
            priority,
        }
    }

    #[test]
    fn admits_within_budget() {
        // block_size 4, 100 blocks. A 4-token prompt needs 1 block.
        let mut b = ContinuousBatcher::new(4, 100, 8).expect("batcher");
        b.submit(&req(1, 4, 2, 0));
        b.submit(&req(2, 4, 2, 0));
        let plan = b.step();
        assert_eq!(plan.admitted.len(), 2, "both fit");
        assert_eq!(b.n_running(), 2);
    }

    #[test]
    fn never_exceeds_block_capacity() {
        // 3 blocks only; each 4-token prompt needs 1 block; cap batch at 8.
        let mut b = ContinuousBatcher::new(4, 3, 8).expect("batcher");
        for i in 0..10 {
            b.submit(&req(i, 4, 5, 0));
        }
        // Step many times; invariant: free_blocks ∈ [0, total].
        for _ in 0..50 {
            let plan = b.step();
            assert!(plan.free_blocks <= 3);
            // used = total - free must be ≥ 0 implicitly.
        }
    }

    #[test]
    fn priority_admitted_first() {
        let mut b = ContinuousBatcher::new(4, 1, 1).expect("batcher"); // only 1 block, batch 1
        b.submit(&req(1, 4, 1, 0)); // low priority
        b.submit(&req(2, 4, 1, 5)); // high priority — should jump the queue
        let plan = b.step();
        assert_eq!(
            plan.admitted,
            vec![2],
            "high-priority request admitted first"
        );
    }

    #[test]
    fn sequence_finishes_after_max_new_tokens() {
        let mut b = ContinuousBatcher::new(8, 100, 8).expect("batcher");
        b.submit(&req(1, 2, 3, 0));
        // Iter 1: admit (prefill). Iters 2..: decode.
        let p1 = b.step();
        assert_eq!(p1.admitted, vec![1]);
        // It takes 3 decode steps to finish (generated 1,2,3).
        let mut total_decoded = 0;
        let mut finished_at = None;
        for it in 0..5 {
            let p = b.step();
            total_decoded += p.decoded.len();
            if p.finished.contains(&1) {
                finished_at = Some(it);
                break;
            }
        }
        assert_eq!(finished_at, Some(2), "finishes on the 3rd decode step");
        assert_eq!(total_decoded, 3);
    }

    #[test]
    fn blocks_freed_on_finish() {
        let mut b = ContinuousBatcher::new(8, 10, 8).expect("batcher");
        b.submit(&req(1, 2, 1, 0));
        b.step(); // admit, 1 block used
        assert_eq!(b.free_blocks(), 9);
        b.step(); // decode 1 token → finishes → frees block
        assert_eq!(b.free_blocks(), 10, "block returned on finish");
    }

    #[test]
    fn run_to_completion_drains_all() {
        let mut b = ContinuousBatcher::new(4, 5, 4).expect("batcher");
        for i in 0..6 {
            b.submit(&req(i, 4, 3, 0));
        }
        let iters = b.run_to_completion().expect("completes");
        assert_eq!(b.n_running(), 0);
        assert_eq!(b.n_waiting(), 0);
        assert!(iters > 0);
    }

    #[test]
    fn growing_sequence_allocates_new_block() {
        // block_size 2; prompt 2 tokens = 1 block. After 2 generated tokens
        // (total 4) it needs 2 blocks.
        let mut b = ContinuousBatcher::new(2, 10, 4).expect("batcher");
        b.submit(&req(1, 2, 4, 0));
        b.step(); // admit: 1 block
        assert_eq!(b.free_blocks(), 9);
        b.step(); // generated 1 → total 3 → ceil(3/2)=2 blocks
        assert_eq!(b.free_blocks(), 8, "crossed a block boundary");
    }

    #[test]
    fn preemption_when_out_of_memory() {
        // 1 block, block_size 2. Seq 1 prompt 2 (1 block). It will need a 2nd
        // block at total=3 but none exists → preempted back to waiting.
        let mut b = ContinuousBatcher::new(2, 1, 4).expect("batcher");
        b.submit(&req(1, 2, 5, 0));
        b.step(); // admit
        assert_eq!(b.n_running(), 1);
        let plan = b.step(); // tries to grow → OOM → preempt
        assert!(plan.decoded.is_empty(), "decode rolled back");
        assert_eq!(b.n_running(), 0, "preempted");
        assert_eq!(b.n_waiting(), 1, "requeued");
        assert_eq!(b.free_blocks(), 1, "blocks freed on preempt");
    }

    #[test]
    fn impossible_request_deadlocks_reported() {
        // A request needing more blocks than exist can never run.
        let mut b = ContinuousBatcher::new(2, 1, 4).expect("batcher");
        b.submit(&req(1, 6, 1, 0)); // 6 tokens → 3 blocks, only 1 exists
        let err = b.run_to_completion();
        assert!(matches!(err, Err(DistInferError::Internal(_))));
    }

    #[test]
    fn state_of_tracks_lifecycle() {
        let mut b = ContinuousBatcher::new(8, 10, 4).expect("batcher");
        b.submit(&req(7, 2, 1, 0));
        assert_eq!(b.state_of(7), Some(SeqState::Waiting));
        b.step();
        assert_eq!(b.state_of(7), Some(SeqState::Running));
        assert_eq!(b.state_of(999), None);
    }

    #[test]
    fn zero_params_error() {
        assert!(ContinuousBatcher::new(0, 10, 4).is_err());
        assert!(ContinuousBatcher::new(4, 0, 4).is_err());
        assert!(ContinuousBatcher::new(4, 10, 0).is_err());
    }

    #[test]
    fn batch_width_cap_respected() {
        let mut b = ContinuousBatcher::new(8, 100, 2).expect("batcher"); // cap 2
        for i in 0..5 {
            b.submit(&req(i, 2, 5, 0));
        }
        b.step();
        assert!(b.n_running() <= 2, "batch width cap enforced");
    }
}

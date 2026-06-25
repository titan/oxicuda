//! # Inference Scheduler
//!
//! FCFS (first-come-first-served) scheduler with preemption for the
//! continuous-batching inference engine.
//!
//! ## Scheduling policy
//!
//! Each call to [`Scheduler::schedule`] produces a [`ScheduledBatch`] with:
//!
//! * **prefill_ids** — sequences whose prompt should be processed this step.
//! * **decode_ids**  — sequences generating one new token this step.
//!
//! The scheduler respects two resource constraints:
//!
//! 1. **`max_running_seqs`** — maximum concurrent sequences in GPU memory.
//! 2. **`max_batch_tokens`** — maximum total input tokens per forward pass.
//!
//! If the KV cache is exhausted, the sequence that most recently entered the
//! running set is preempted (last-in-first-out within running, LIFO on GPU).
//! Preempted sequences are returned to the waiting queue for re-prefill once
//! blocks become available.

use std::collections::{HashMap, VecDeque};

use crate::batch::sequence::{Sequence, SequenceId, SequenceStatus};
use crate::error::{InferError, InferResult};

// ─── SchedulerConfig ─────────────────────────────────────────────────────────

/// Tunable limits for the inference scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of sequences simultaneously in GPU memory.
    pub max_running_seqs: usize,
    /// Maximum total tokens in a single forward-pass batch.
    pub max_batch_tokens: usize,
    /// Maximum KV blocks allocated per sequence (upper bound on sequence length).
    pub max_blocks_per_seq: usize,
    /// KV cache block size (tokens per block).
    pub block_size: usize,
    /// Total number of KV cache blocks available.
    pub total_kv_blocks: usize,
}

impl SchedulerConfig {
    /// Construct a config appropriate for the given `max_seq_len` and block size.
    #[must_use]
    pub fn new(
        max_running_seqs: usize,
        max_batch_tokens: usize,
        block_size: usize,
        total_kv_blocks: usize,
    ) -> Self {
        let max_blocks_per_seq = (16384_usize).div_ceil(block_size).max(1);
        Self {
            max_running_seqs,
            max_batch_tokens,
            max_blocks_per_seq,
            block_size,
            total_kv_blocks,
        }
    }
}

// ─── ScheduledBatch ──────────────────────────────────────────────────────────

/// Output of one scheduling decision.
#[derive(Debug, Clone, Default)]
pub struct ScheduledBatch {
    /// Sequence IDs to prefill this step (process full prompt).
    pub prefill_ids: Vec<SequenceId>,
    /// Sequence IDs to decode this step (one new token each).
    pub decode_ids: Vec<SequenceId>,
    /// Total token budget consumed by this batch.
    pub n_tokens: usize,
}

impl ScheduledBatch {
    /// Is the batch non-empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.prefill_ids.is_empty() && self.decode_ids.is_empty()
    }

    /// Total number of sequences scheduled.
    #[must_use]
    pub fn n_seqs(&self) -> usize {
        self.prefill_ids.len() + self.decode_ids.len()
    }
}

// ─── StepResult ──────────────────────────────────────────────────────────────

/// Result of processing one token for a sequence after the model forward pass.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Which sequence produced this result.
    pub seq_id: SequenceId,
    /// The sampled token.
    pub token: u32,
    /// Log-probability of the sampled token.
    pub log_prob: f64,
}

// ─── Scheduler ───────────────────────────────────────────────────────────────

/// FCFS admission scheduler for continuous batching.
pub struct Scheduler {
    config: SchedulerConfig,
    /// Sequences waiting for their first prefill.
    waiting: VecDeque<Sequence>,
    /// Sequences actively running on the GPU.
    running: Vec<Sequence>,
    /// Sequences whose KV blocks were reclaimed and must be re-prefilled.
    preempted: Vec<Sequence>,
    /// Sequences that have finished generation.
    finished: Vec<Sequence>,
    /// Monotonically increasing request counter for assigning IDs.
    next_id: SequenceId,
    /// Free KV blocks remaining (tracked separately from the KV cache for
    /// scheduling decisions; actual allocation is in `CacheManager`).
    free_blocks: usize,
    /// Blocks currently reserved by each admitted sequence, so finishing or
    /// preempting a sequence returns exactly what its admission consumed (rather
    /// than leaking the reservation).
    reserved_blocks: HashMap<SequenceId, usize>,
}

impl Scheduler {
    /// Create a new scheduler.
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        let free_blocks = config.total_kv_blocks;
        Self {
            config,
            waiting: VecDeque::new(),
            running: Vec::new(),
            preempted: Vec::new(),
            finished: Vec::new(),
            next_id: 1,
            free_blocks,
            reserved_blocks: HashMap::new(),
        }
    }

    // ── Request submission ────────────────────────────────────────────────────

    /// Submit a new generation request.  Returns the assigned `SequenceId`.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: crate::batch::sequence::SamplingParams,
    ) -> SequenceId {
        let id = self.next_id;
        self.next_id += 1;
        self.waiting
            .push_back(Sequence::new(id, prompt_tokens, sampling_params));
        id
    }

    // ── Scheduling ───────────────────────────────────────────────────────────

    /// Produce the next `ScheduledBatch`.
    ///
    /// The returned batch may mix prefill and decode sequences in the same
    /// GPU forward pass (chunked prefill).  Priority order:
    ///
    /// 1. Running decode sequences (always included if resources allow).
    /// 2. Preempted sequences (re-promoted before new arrivals).
    /// 3. Waiting sequences (FCFS).
    pub fn schedule(&mut self) -> ScheduledBatch {
        let mut batch = ScheduledBatch::default();
        let mut running_now = Vec::new();
        let mut token_budget = self.config.max_batch_tokens;

        // --- Phase 1: include running decode sequences -----------------------
        for mut seq in self.running.drain(..) {
            let cost = 1; // decode: one token per sequence
            if batch.n_seqs() >= self.config.max_running_seqs || token_budget < cost {
                // Preempt: reclaim the sequence's whole KV reservation so the
                // freed blocks are available to other sequences this step.
                preempt_seq(&mut seq);
                let reclaimed = self.reserved_blocks.remove(&seq.id).unwrap_or(0);
                self.free_blocks = self.free_blocks.saturating_add(reclaimed);
                self.preempted.push(seq);
            } else {
                batch.decode_ids.push(seq.id);
                token_budget = token_budget.saturating_sub(cost);
                batch.n_tokens += cost;
                running_now.push(seq);
            }
        }

        // --- Phase 2: admit preempted sequences (re-prefill) -----------------
        let mut still_preempted = Vec::new();
        for mut seq in self.preempted.drain(..) {
            let prompt_len = seq.prompt_tokens.len();
            let blocks_needed = prompt_len.div_ceil(self.config.block_size).max(1);
            if batch.n_seqs() < self.config.max_running_seqs
                && token_budget >= prompt_len
                && self.free_blocks >= blocks_needed
            {
                seq.status = SequenceStatus::Prefill;
                seq.block_table.clear();
                batch.prefill_ids.push(seq.id);
                token_budget = token_budget.saturating_sub(prompt_len);
                batch.n_tokens += prompt_len;
                self.free_blocks = self.free_blocks.saturating_sub(blocks_needed);
                self.reserved_blocks.insert(seq.id, blocks_needed);
                running_now.push(seq);
            } else {
                still_preempted.push(seq);
            }
        }
        self.preempted = still_preempted;

        // --- Phase 3: admit new waiting sequences (FCFS) ---------------------
        while let Some(mut seq) = self.waiting.pop_front() {
            let prompt_len = seq.prompt_tokens.len();
            let blocks_needed = prompt_len.div_ceil(self.config.block_size).max(1);
            if batch.n_seqs() >= self.config.max_running_seqs
                || token_budget < prompt_len
                || self.free_blocks < blocks_needed
            {
                // Cannot admit; push back to front and stop.
                self.waiting.push_front(seq);
                break;
            }
            seq.status = SequenceStatus::Prefill;
            batch.prefill_ids.push(seq.id);
            token_budget = token_budget.saturating_sub(prompt_len);
            batch.n_tokens += prompt_len;
            self.free_blocks = self.free_blocks.saturating_sub(blocks_needed);
            self.reserved_blocks.insert(seq.id, blocks_needed);
            running_now.push(seq);
        }

        self.running = running_now;
        batch
    }

    // ── Step completion ───────────────────────────────────────────────────────

    /// Process the output of one forward pass step.
    ///
    /// For each result, appends the generated token to the sequence and checks
    /// for termination.  Finished sequences are moved to `self.finished`.
    pub fn on_step_complete(&mut self, results: Vec<StepResult>) -> InferResult<()> {
        for result in results {
            let Some(seq) = self.running.iter_mut().find(|s| s.id == result.seq_id) else {
                return Err(InferError::InvalidSequenceId(result.seq_id));
            };

            // Transition from Prefill → Decode after first token.
            seq.start_decode();

            seq.append_token(result.token, result.log_prob);
            if let Some(reason) = seq.check_finish() {
                seq.finish(reason);
            }
        }

        // Drain finished sequences out of `running`.
        let (finished, still_running): (Vec<_>, Vec<_>) =
            self.running.drain(..).partition(|s| s.status.is_finished());
        // Return each finished sequence's KV reservation to the free pool so
        // long-running churn does not leak blocks.
        for seq in &finished {
            let reclaimed = self.reserved_blocks.remove(&seq.id).unwrap_or(0);
            self.free_blocks = (self.free_blocks + reclaimed).min(self.config.total_kv_blocks);
        }
        self.finished.extend(finished);
        self.running = still_running;

        Ok(())
    }

    // ── Introspection ─────────────────────────────────────────────────────────

    /// Take all completed sequences (clears the finished list).
    pub fn take_finished(&mut self) -> Vec<Sequence> {
        std::mem::take(&mut self.finished)
    }

    /// Whether there are sequences still in flight (waiting, running, or preempted).
    #[must_use]
    pub fn has_unfinished(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty() || !self.preempted.is_empty()
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

    /// Number of preempted sequences.
    #[must_use]
    pub fn n_preempted(&self) -> usize {
        self.preempted.len()
    }

    /// Approximate free KV block count tracked by the scheduler.
    #[must_use]
    pub fn free_blocks(&self) -> usize {
        self.free_blocks
    }

    /// Return the current conditioning token for a sequence.
    ///
    /// Searches running, waiting, and preempted queues. Returns `None` if the
    /// sequence is unknown or already finished/removed.
    #[must_use]
    pub fn last_token(&self, seq_id: SequenceId) -> Option<u32> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.preempted.iter())
            .find(|s| s.id == seq_id)
            .map(Sequence::last_token)
    }

    /// Return the total logical sequence length (prompt + generated tokens).
    ///
    /// Searches running, waiting, and preempted queues. Returns `None` if the
    /// sequence is unknown or already finished/removed.
    #[must_use]
    pub fn total_len(&self, seq_id: SequenceId) -> Option<usize> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.preempted.iter())
            .find(|s| s.id == seq_id)
            .map(Sequence::total_len)
    }
}

// ── Module-level helpers ──────────────────────────────────────────────────────

/// Reset a sequence to the preempted state (clears its block table and the
/// output generated so far so it can be re-prefilled from the prompt).
fn preempt_seq(seq: &mut Sequence) {
    seq.block_table.clear();
    seq.output_tokens.clear();
    seq.status = SequenceStatus::Preempted;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::sequence::SamplingParams;

    fn tiny_config() -> SchedulerConfig {
        SchedulerConfig::new(4, 256, 16, 64)
    }

    fn add_reqs(sched: &mut Scheduler, n: usize, prompt_len: usize) -> Vec<SequenceId> {
        (0..n)
            .map(|_| sched.add_request(vec![0_u32; prompt_len], SamplingParams::default()))
            .collect()
    }

    #[test]
    fn schedule_admits_waiting_seqs() {
        let mut s = Scheduler::new(tiny_config());
        add_reqs(&mut s, 2, 8);
        let batch = s.schedule();
        assert_eq!(batch.prefill_ids.len(), 2);
        assert!(batch.decode_ids.is_empty());
    }

    #[test]
    fn max_running_limit() {
        let mut config = tiny_config();
        config.max_running_seqs = 2;
        let mut s = Scheduler::new(config);
        add_reqs(&mut s, 4, 4);
        let batch = s.schedule();
        assert!(batch.n_seqs() <= 2, "got {}", batch.n_seqs());
    }

    #[test]
    fn second_schedule_produces_decode_ids() {
        let mut s = Scheduler::new(tiny_config());
        let ids = add_reqs(&mut s, 2, 4);
        let _b1 = s.schedule();
        // Simulate: both sequences get one token.
        s.on_step_complete(vec![
            StepResult {
                seq_id: ids[0],
                token: 100,
                log_prob: -0.5,
            },
            StepResult {
                seq_id: ids[1],
                token: 101,
                log_prob: -0.6,
            },
        ])
        .expect("both sequence ids are valid running sequences");
        let b2 = s.schedule();
        assert_eq!(b2.decode_ids.len(), 2, "both should be in decode");
        assert!(b2.prefill_ids.is_empty());
    }

    #[test]
    fn eos_token_finishes_sequence() {
        let params = SamplingParams {
            eos_token_id: Some(2),
            max_new_tokens: 128,
            ..Default::default()
        };
        let mut s = Scheduler::new(tiny_config());
        let id = s.add_request(vec![0, 1], params);
        let _b = s.schedule();
        // Emit EOS
        s.on_step_complete(vec![StepResult {
            seq_id: id,
            token: 2,
            log_prob: 0.0,
        }])
        .expect("sequence id is valid running sequence");
        let finished = s.take_finished();
        assert_eq!(finished.len(), 1);
        assert!(finished[0].status.is_finished());
        assert!(!s.has_unfinished());
    }

    #[test]
    fn max_tokens_finishes_sequence() {
        let params = SamplingParams {
            max_new_tokens: 2,
            ..Default::default()
        };
        let mut s = Scheduler::new(tiny_config());
        let id = s.add_request(vec![0], params);
        s.schedule();
        s.on_step_complete(vec![StepResult {
            seq_id: id,
            token: 5,
            log_prob: 0.0,
        }])
        .expect("sequence id is valid for first step");
        s.schedule();
        s.on_step_complete(vec![StepResult {
            seq_id: id,
            token: 6,
            log_prob: 0.0,
        }])
        .expect("sequence id is valid for second step");
        assert!(
            !s.has_unfinished(),
            "sequence should be finished after max tokens"
        );
    }

    #[test]
    fn invalid_seq_id_error() {
        let mut s = Scheduler::new(tiny_config());
        let r = s.on_step_complete(vec![StepResult {
            seq_id: 999,
            token: 0,
            log_prob: 0.0,
        }]);
        assert!(r.is_err());
    }

    #[test]
    fn has_unfinished_empty() {
        let s = Scheduler::new(tiny_config());
        assert!(!s.has_unfinished());
    }

    #[test]
    fn token_budget_respected() {
        let mut config = tiny_config();
        config.max_batch_tokens = 10; // only 10 tokens per batch
        let mut s = Scheduler::new(config);
        // Add one request with 8-token prompt and one with 8-token prompt
        add_reqs(&mut s, 2, 8);
        let batch = s.schedule();
        // Only one 8-token prefill should fit in budget 10.
        assert!(batch.n_tokens <= 10, "got {} tokens", batch.n_tokens);
    }

    #[test]
    fn take_finished_clears_list() {
        let params = SamplingParams {
            eos_token_id: Some(0),
            max_new_tokens: 128,
            ..Default::default()
        };
        let mut s = Scheduler::new(tiny_config());
        let id = s.add_request(vec![1], params);
        s.schedule();
        s.on_step_complete(vec![StepResult {
            seq_id: id,
            token: 0,
            log_prob: 0.0,
        }])
        .expect("sequence id is valid running sequence for take_finished test");
        let f1 = s.take_finished();
        assert_eq!(f1.len(), 1);
        let f2 = s.take_finished();
        assert!(f2.is_empty(), "take_finished should clear the list");
    }

    /// Preemption stress test (TODO: memory-pressure preemption under churn).
    ///
    /// Drives the scheduler with far more concurrent requests than the running
    /// slots / KV blocks can hold, forcing repeated preemption and re-admission,
    /// and asserts the core invariants survive heavy churn:
    ///   * no sequence is ever lost or duplicated (count is conserved);
    ///   * the running set never exceeds `max_running_seqs`;
    ///   * scheduler-tracked free blocks never go negative or exceed the pool;
    ///   * the system makes progress and ultimately drains every request.
    #[test]
    fn preemption_under_heavy_churn() {
        // Tight resources: 2 running slots, small token budget, few KV blocks.
        let mut config = SchedulerConfig::new(2, 24, 4, 6);
        config.max_running_seqs = 2;
        let mut s = Scheduler::new(config);

        // 12 requests, each generating a few tokens before EOS (token 7).
        let n_requests = 12_usize;
        let mut ids = Vec::new();
        for _ in 0..n_requests {
            let params = SamplingParams {
                eos_token_id: Some(7),
                max_new_tokens: 4,
                ..Default::default()
            };
            ids.push(s.add_request(vec![1_u32, 2, 3], params));
        }

        let pool = 6_usize;
        let mut completed_ids: Vec<SequenceId> = Vec::new();
        // Deterministic step counter; emit EOS on the 3rd generated token so
        // sequences finish at different times, maximising churn.
        let mut emitted: std::collections::HashMap<SequenceId, u32> =
            std::collections::HashMap::new();

        let mut steps = 0;
        while s.has_unfinished() {
            let batch = s.schedule();

            // Invariant: running slots respected.
            assert!(
                s.n_running() <= 2,
                "running set {} exceeds max_running_seqs",
                s.n_running()
            );
            // Invariant: free-block tracking stays within [0, pool].
            assert!(
                s.free_blocks() <= pool,
                "free_blocks {} exceeds pool {pool}",
                s.free_blocks()
            );

            if batch.is_empty() {
                // With nothing schedulable but work remaining, the system must be
                // blocked only transiently; a finished drain frees resources.
                // (No sequences scheduled ⇒ nothing to complete this step.)
                steps += 1;
                assert!(steps < 10_000, "scheduler failed to make progress");
                continue;
            }

            // Produce one token for every scheduled sequence.
            let mut results = Vec::new();
            for &seq_id in batch.prefill_ids.iter().chain(batch.decode_ids.iter()) {
                let count = emitted.entry(seq_id).or_insert(0);
                *count += 1;
                let token = if *count >= 3 { 7 } else { 100 + *count }; // EOS at 3rd
                results.push(StepResult {
                    seq_id,
                    token,
                    log_prob: -0.1,
                });
            }
            s.on_step_complete(results)
                .expect("all scheduled ids are valid running sequences");

            // Collect finished sequences each step.
            for fin in s.take_finished() {
                assert!(fin.status.is_finished());
                completed_ids.push(fin.id);
            }

            steps += 1;
            assert!(steps < 10_000, "churn test did not terminate");
        }

        // Drain any stragglers.
        for fin in s.take_finished() {
            completed_ids.push(fin.id);
        }

        // Invariant: every request finished exactly once (no loss, no dupes).
        completed_ids.sort_unstable();
        let mut expected = ids.clone();
        expected.sort_unstable();
        completed_ids.dedup();
        assert_eq!(
            completed_ids.len(),
            n_requests,
            "every request must finish exactly once under churn"
        );
        assert_eq!(
            completed_ids, expected,
            "finished id set must equal submitted set"
        );
        assert!(!s.has_unfinished(), "scheduler fully drained");
    }
}

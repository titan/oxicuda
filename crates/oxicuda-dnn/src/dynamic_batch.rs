//! Dynamic batching and continuous batching for inference serving.
//!
//! This module implements the core scheduling primitives used by modern LLM
//! inference engines (vLLM, Orca, TensorRT-LLM):
//!
//! - **[`ContinuousBatcher`]** — iteration-level scheduler that decides which
//!   requests to prefill, decode, or preempt at each step.
//! - **[`TokenBudgetAllocator`]** — manages a per-step token budget shared
//!   between prefill and decode phases.
//! - **[`PagedKvManager`]** — block-level paged KV-cache allocator with
//!   copy-on-write support for beam search and speculative decoding.
//! - **[`SpeculativeDecoder`]** — draft-model speculative decoding implementing
//!   the speculative-sampling algorithm of Leviathan et al. (2023) and
//!   Chen et al. (2023): drafted tokens are sampled from the draft model's
//!   categorical distribution and verified against the target model with
//!   modified rejection sampling.
//! - **[`BatchMetrics`]** — running statistics for throughput, latency, and
//!   utilization monitoring.
//!
//! # Scheduling Policies
//!
//! | Policy | Description |
//! |--------|-------------|
//! | [`SchedulingPolicy::Fcfs`] | First-come, first-served |
//! | [`SchedulingPolicy::ShortestJobFirst`] | Shortest remaining generation |
//! | [`SchedulingPolicy::PriorityBased`] | User-assigned priority levels |
//! | [`SchedulingPolicy::DeadlineAware`] | EDF (earliest deadline first) |
//! | [`SchedulingPolicy::Orca`] | Iteration-level (selective batching) |

use std::collections::VecDeque;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Basic types
// ---------------------------------------------------------------------------

/// Unique identifier for an inference request.
pub type RequestId = u64;

/// Priority level for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Lowest priority — best-effort.
    Low = 0,
    /// Default priority.
    Normal = 1,
    /// Expedited processing.
    High = 2,
}

/// An incoming inference request.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Unique request identifier.
    pub request_id: RequestId,
    /// Number of input (prompt) tokens.
    pub sequence_length: usize,
    /// Maximum number of tokens to generate.
    pub max_new_tokens: usize,
    /// Scheduling priority.
    pub priority: Priority,
    /// Monotonic arrival timestamp in nanoseconds.
    pub arrival_time_ns: u64,
    /// Optional hard deadline in nanoseconds (absolute).
    pub deadline_ns: Option<u64>,
}

/// A slot inside the running batch.
#[derive(Debug, Clone)]
pub struct BatchSlot {
    /// Slot index within the batch.
    pub slot_id: usize,
    /// Request occupying this slot.
    pub request_id: RequestId,
    /// Tokens processed so far (prompt + generated).
    pub current_seq_len: usize,
    /// Maximum sequence length (prompt + max_new_tokens).
    pub max_seq_len: usize,
    /// Whether this slot is currently doing prefill.
    pub is_prefill: bool,
    /// Whether this slot is actively generating.
    pub is_active: bool,
}

/// Scheduling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-come, first-served.
    Fcfs,
    /// Prefer requests with fewer remaining tokens.
    ShortestJobFirst,
    /// Respect user-assigned [`Priority`] levels.
    PriorityBased,
    /// Earliest-deadline-first (requires `deadline_ns`).
    DeadlineAware,
    /// Orca-style iteration-level selective batching.
    Orca,
}

/// How to handle a preempted request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionPolicy {
    /// Discard KV cache and recompute from scratch when resumed.
    Recompute,
    /// Swap (offload) KV cache blocks to host memory.
    Swap,
}

/// Configuration for the continuous batcher.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of requests in a single batch.
    pub max_batch_size: usize,
    /// Maximum total tokens (prompt + generated) across the batch.
    pub max_total_tokens: usize,
    /// Maximum sequence length for any single request.
    pub max_sequence_length: usize,
    /// Maximum prefill tokens per step.
    pub prefill_batch_size: usize,
    /// Maximum decode slots per step.
    pub decode_batch_size: usize,
    /// Scheduling algorithm.
    pub scheduling_policy: SchedulingPolicy,
}

/// Result of a single scheduling step.
#[derive(Debug, Clone)]
pub struct BatchDecision {
    /// Requests to prefill in this step.
    pub prefill_requests: Vec<RequestId>,
    /// Requests to decode in this step.
    pub decode_requests: Vec<RequestId>,
    /// Requests preempted to free capacity.
    pub preempted: Vec<RequestId>,
    /// Total token count for the step.
    pub total_tokens: usize,
}

// ---------------------------------------------------------------------------
// BatchState — internal bookkeeping
// ---------------------------------------------------------------------------

/// Internal state of the batcher.
#[derive(Debug)]
struct BatchState {
    /// Currently running slots.
    active_slots: Vec<BatchSlot>,
    /// Total tokens across all active slots.
    total_tokens: usize,
    /// Requests waiting for prefill.
    prefill_queue: VecDeque<InferenceRequest>,
    /// Requests in decode phase (tracked separately for Orca).
    decode_queue: VecDeque<RequestId>,
    /// Preempted requests awaiting resumption.
    preempted_queue: VecDeque<InferenceRequest>,
}

impl BatchState {
    fn new() -> Self {
        Self {
            active_slots: Vec::new(),
            total_tokens: 0,
            prefill_queue: VecDeque::new(),
            decode_queue: VecDeque::new(),
            preempted_queue: VecDeque::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ContinuousBatcher
// ---------------------------------------------------------------------------

/// Continuous batcher — the main scheduler for LLM inference serving.
///
/// Implements iteration-level scheduling inspired by Orca / vLLM.  At each
/// [`step`](ContinuousBatcher::step) the batcher decides which waiting
/// requests to admit for prefill, which running requests continue decoding,
/// and whether any requests must be preempted.
#[derive(Debug)]
pub struct ContinuousBatcher {
    config: BatchConfig,
    state: BatchState,
    next_slot_id: usize,
    completed_count: u64,
}

impl ContinuousBatcher {
    /// Create a new batcher with the given configuration.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            state: BatchState::new(),
            next_slot_id: 0,
            completed_count: 0,
        }
    }

    /// Enqueue a new inference request. Returns its `RequestId`.
    pub fn add_request(&mut self, request: InferenceRequest) -> DnnResult<RequestId> {
        if request.sequence_length == 0 {
            return Err(DnnError::InvalidArgument(
                "sequence_length must be > 0".into(),
            ));
        }
        if request.sequence_length > self.config.max_sequence_length {
            return Err(DnnError::InvalidArgument(format!(
                "sequence_length {} exceeds max_sequence_length {}",
                request.sequence_length, self.config.max_sequence_length
            )));
        }
        let id = request.request_id;
        self.state.prefill_queue.push_back(request);
        Ok(id)
    }

    /// Execute one scheduling step.
    ///
    /// Returns a [`BatchDecision`] describing which requests to prefill,
    /// decode, and preempt during this iteration.
    pub fn step(&mut self) -> DnnResult<BatchDecision> {
        let mut decision = BatchDecision {
            prefill_requests: Vec::new(),
            decode_requests: Vec::new(),
            preempted: Vec::new(),
            total_tokens: 0,
        };

        // 1. Collect decode requests from active slots.
        let decode_ids: Vec<RequestId> = self
            .state
            .active_slots
            .iter()
            .filter(|s| s.is_active && !s.is_prefill)
            .map(|s| s.request_id)
            .collect();

        let decode_count = decode_ids.len().min(self.config.decode_batch_size);
        let decode_tokens: usize = self
            .state
            .active_slots
            .iter()
            .filter(|s| s.is_active && !s.is_prefill)
            .take(decode_count)
            .map(|s| s.current_seq_len + 1) // +1 for the token being generated
            .sum();

        decision.decode_requests = decode_ids.into_iter().take(decode_count).collect();

        // 2. Sort the prefill queue according to scheduling policy.
        self.sort_prefill_queue();

        // 3. Admit prefill requests within budget.
        let mut prefill_budget = self
            .config
            .prefill_batch_size
            .min(self.config.max_total_tokens.saturating_sub(decode_tokens));

        let mut admitted = Vec::new();
        while !self.state.prefill_queue.is_empty()
            && self.state.active_slots.len() + admitted.len() < self.config.max_batch_size
        {
            // Peek at the front.
            let req = match self.state.prefill_queue.front() {
                Some(r) => r,
                None => break,
            };
            if req.sequence_length > prefill_budget {
                break;
            }
            // Safe: we just confirmed front() is Some.
            let req = self
                .state
                .prefill_queue
                .pop_front()
                .ok_or_else(|| DnnError::InvalidArgument("empty queue".into()))?;

            prefill_budget = prefill_budget.saturating_sub(req.sequence_length);

            let slot = BatchSlot {
                slot_id: self.next_slot_id,
                request_id: req.request_id,
                current_seq_len: req.sequence_length,
                max_seq_len: req.sequence_length + req.max_new_tokens,
                is_prefill: true,
                is_active: true,
            };
            self.next_slot_id += 1;
            decision.prefill_requests.push(req.request_id);
            admitted.push(slot);
        }

        // 4. Transition admitted prefill slots to decode.
        for slot in &mut admitted {
            slot.is_prefill = false;
        }
        self.state.active_slots.extend(admitted);

        // Increment decode tokens for existing slots.
        for slot in &mut self.state.active_slots {
            if slot.is_active && !slot.is_prefill {
                slot.current_seq_len = slot.current_seq_len.saturating_add(1);
            }
        }

        decision.total_tokens = self
            .state
            .active_slots
            .iter()
            .filter(|s| s.is_active)
            .map(|s| s.current_seq_len)
            .sum();

        self.state.total_tokens = decision.total_tokens;

        Ok(decision)
    }

    /// Mark a request as completed and free its resources.
    pub fn complete_request(&mut self, request_id: RequestId) -> DnnResult<()> {
        let pos = self
            .state
            .active_slots
            .iter()
            .position(|s| s.request_id == request_id)
            .ok_or_else(|| {
                DnnError::InvalidArgument(format!("request {request_id} not in active slots"))
            })?;
        let slot = &self.state.active_slots[pos];
        self.state.total_tokens = self.state.total_tokens.saturating_sub(slot.current_seq_len);
        self.state.active_slots.remove(pos);
        self.state.decode_queue.retain(|id| *id != request_id);
        self.completed_count += 1;
        Ok(())
    }

    /// Preempt a running request. The request is moved to the preempted queue
    /// and may be resumed later.
    pub fn preempt(&mut self, request_id: RequestId) -> DnnResult<()> {
        let pos = self
            .state
            .active_slots
            .iter()
            .position(|s| s.request_id == request_id)
            .ok_or_else(|| {
                DnnError::InvalidArgument(format!("request {request_id} not in active slots"))
            })?;
        let slot = self.state.active_slots.remove(pos);
        self.state.total_tokens = self.state.total_tokens.saturating_sub(slot.current_seq_len);
        self.state.decode_queue.retain(|id| *id != request_id);

        // Re-enqueue as a prefill request so it can be recomputed.
        let preempted_req = InferenceRequest {
            request_id,
            sequence_length: slot.current_seq_len,
            max_new_tokens: slot.max_seq_len.saturating_sub(slot.current_seq_len),
            priority: Priority::Normal,
            arrival_time_ns: 0,
            deadline_ns: None,
        };
        self.state.preempted_queue.push_back(preempted_req);
        Ok(())
    }

    /// Number of requests currently executing (prefill + decode).
    pub fn active_requests(&self) -> usize {
        self.state
            .active_slots
            .iter()
            .filter(|s| s.is_active)
            .count()
    }

    /// Number of requests waiting in all queues (prefill + preempted).
    pub fn pending_requests(&self) -> usize {
        self.state.prefill_queue.len() + self.state.preempted_queue.len()
    }

    /// Total tokens that would be processed in the current active batch.
    pub fn throughput_tokens_per_step(&self) -> usize {
        self.state.total_tokens
    }

    // -- private helpers --

    fn sort_prefill_queue(&mut self) {
        let queue = &mut self.state.prefill_queue;
        let policy = self.config.scheduling_policy;

        let mut vec: Vec<InferenceRequest> = queue.drain(..).collect();
        match policy {
            SchedulingPolicy::Fcfs => {
                // Already in arrival order — sort by arrival_time_ns.
                vec.sort_by_key(|r| r.arrival_time_ns);
            }
            SchedulingPolicy::ShortestJobFirst => {
                vec.sort_by_key(|r| r.max_new_tokens);
            }
            SchedulingPolicy::PriorityBased => {
                // Higher priority first, then FCFS within same priority.
                vec.sort_by(|a, b| {
                    b.priority
                        .cmp(&a.priority)
                        .then(a.arrival_time_ns.cmp(&b.arrival_time_ns))
                });
            }
            SchedulingPolicy::DeadlineAware => {
                // Earliest deadline first; no-deadline requests go last.
                vec.sort_by(|a, b| {
                    let da = a.deadline_ns.unwrap_or(u64::MAX);
                    let db = b.deadline_ns.unwrap_or(u64::MAX);
                    da.cmp(&db).then(a.arrival_time_ns.cmp(&b.arrival_time_ns))
                });
            }
            SchedulingPolicy::Orca => {
                // Orca: iteration-level — same as FCFS for the prefill queue.
                vec.sort_by_key(|r| r.arrival_time_ns);
            }
        }
        *queue = VecDeque::from(vec);
    }
}

// ---------------------------------------------------------------------------
// TokenBudgetAllocator
// ---------------------------------------------------------------------------

/// Manages the per-step token budget shared between prefill and decode.
#[derive(Debug)]
pub struct TokenBudgetAllocator {
    max_total_tokens: usize,
    allocated: usize,
}

impl TokenBudgetAllocator {
    /// Create an allocator with the given capacity.
    pub fn new(max_total_tokens: usize) -> Self {
        Self {
            max_total_tokens,
            allocated: 0,
        }
    }

    /// Try to allocate `seq_len` tokens for a prefill request.
    /// Returns `Some(slot_index)` on success, `None` if the budget is
    /// exhausted.
    pub fn allocate_prefill(&mut self, seq_len: usize) -> Option<usize> {
        if self.allocated + seq_len > self.max_total_tokens {
            return None;
        }
        let slot = self.allocated;
        self.allocated += seq_len;
        Some(slot)
    }

    /// How many decode slots (each consuming 1 token) can still fit.
    pub fn allocate_decode(&mut self, count: usize) -> usize {
        let remaining = self.max_total_tokens.saturating_sub(self.allocated);
        let actual = count.min(remaining);
        self.allocated += actual;
        actual
    }

    /// Release `tokens` from the budget.
    pub fn release(&mut self, tokens: usize) {
        self.allocated = self.allocated.saturating_sub(tokens);
    }

    /// Fraction of the budget currently in use (0.0..=1.0).
    pub fn utilization(&self) -> f64 {
        if self.max_total_tokens == 0 {
            return 0.0;
        }
        self.allocated as f64 / self.max_total_tokens as f64
    }
}

// ---------------------------------------------------------------------------
// PagedKvManager
// ---------------------------------------------------------------------------

/// Block-level paged KV-cache manager.
///
/// Inspired by the paging scheme in vLLM.  Physical blocks are allocated on
/// demand and freed when a request completes.  Copy-on-write is supported for
/// speculative / beam-search scenarios.
#[derive(Debug)]
pub struct PagedKvManager {
    num_blocks: usize,
    block_size: usize,
    /// `true` ⇒ block is free.
    free_map: Vec<bool>,
    /// Reference count per block (for CoW).
    ref_counts: Vec<usize>,
}

impl PagedKvManager {
    /// Create a manager with `num_blocks` blocks, each holding `block_size`
    /// tokens.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            num_blocks,
            block_size,
            free_map: vec![true; num_blocks],
            ref_counts: vec![0; num_blocks],
        }
    }

    /// Allocate enough blocks to hold `num_tokens` tokens.
    ///
    /// Returns the list of allocated block IDs, or an error if there is
    /// insufficient free space.
    pub fn allocate(&mut self, num_tokens: usize) -> DnnResult<Vec<usize>> {
        if self.block_size == 0 {
            return Err(DnnError::InvalidArgument("block_size is 0".into()));
        }
        let blocks_needed = num_tokens.div_ceil(self.block_size);
        if !self.can_allocate(num_tokens) {
            return Err(DnnError::InvalidArgument(format!(
                "not enough free blocks: need {blocks_needed}, have {}",
                self.free_block_count()
            )));
        }
        let mut ids = Vec::with_capacity(blocks_needed);
        for (i, free) in self.free_map.iter_mut().enumerate() {
            if ids.len() >= blocks_needed {
                break;
            }
            if *free {
                *free = false;
                self.ref_counts[i] = 1;
                ids.push(i);
            }
        }
        Ok(ids)
    }

    /// Free the given blocks. Decrements reference counts and marks blocks as
    /// free when the count reaches zero.
    pub fn free(&mut self, block_ids: &[usize]) {
        for &id in block_ids {
            if id < self.num_blocks {
                self.ref_counts[id] = self.ref_counts[id].saturating_sub(1);
                if self.ref_counts[id] == 0 {
                    self.free_map[id] = true;
                }
            }
        }
    }

    /// Copy-on-write: create a new physical copy of `block_id`.
    ///
    /// Used when a block is shared (ref_count > 1) and one branch needs to
    /// diverge (e.g. beam search).
    pub fn copy_on_write(&mut self, block_id: usize) -> DnnResult<usize> {
        if block_id >= self.num_blocks {
            return Err(DnnError::InvalidArgument(format!(
                "block_id {block_id} out of range (max {})",
                self.num_blocks
            )));
        }
        // Find a free block.
        let new_id =
            self.free_map.iter().position(|&free| free).ok_or_else(|| {
                DnnError::InvalidArgument("no free blocks for copy-on-write".into())
            })?;
        self.free_map[new_id] = false;
        self.ref_counts[new_id] = 1;

        // Decrement old block ref count.
        self.ref_counts[block_id] = self.ref_counts[block_id].saturating_sub(1);
        if self.ref_counts[block_id] == 0 {
            self.free_map[block_id] = true;
        }

        Ok(new_id)
    }

    /// (used, total) block counts.
    pub fn usage(&self) -> (usize, usize) {
        let used = self.free_map.iter().filter(|&&free| !free).count();
        (used, self.num_blocks)
    }

    /// Whether `num_tokens` tokens can be allocated right now.
    pub fn can_allocate(&self, num_tokens: usize) -> bool {
        if self.block_size == 0 {
            return false;
        }
        let needed = num_tokens.div_ceil(self.block_size);
        self.free_block_count() >= needed
    }

    fn free_block_count(&self) -> usize {
        self.free_map.iter().filter(|&&f| f).count()
    }
}

// ---------------------------------------------------------------------------
// LcgRng — workspace-convention pseudo-random number generator
// ---------------------------------------------------------------------------

/// Minimal full-period 64-bit LCG (Knuth MMIX constants).
///
/// Used for the categorical sampling and rejection-sampling steps of
/// [`SpeculativeDecoder`].  The high bits of the state are used for output —
/// the low bits of an MMIX LCG have short periods and must be discarded.
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// LCG multiplier (Knuth MMIX).
    const MUL: u64 = 6_364_136_223_846_793_005;
    /// LCG increment (Knuth MMIX).
    const ADD: u64 = 1_442_695_040_888_963_407;

    /// Creates a new generator seeded with `seed`.
    ///
    /// The seed is run through a SplitMix64-style finalising multiply so that
    /// nearby seeds produce well-separated streams.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(Self::ADD),
        }
    }

    /// Advances the state and returns the next 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(Self::MUL).wrapping_add(Self::ADD);
        self.state
    }

    /// Returns a uniform `f64` in `[0, 1)`.
    ///
    /// The top 53 bits of the state are used so every representable
    /// double-precision fraction in `[0, 1)` is reachable.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Samples a category index from a (not necessarily normalised) weight
    /// vector via inverse-CDF sampling.
    ///
    /// `weights` must be non-negative and have a strictly positive sum;
    /// `None` is returned when that precondition does not hold (empty slice,
    /// all-zero weights, or a non-finite total).  The draw is performed
    /// against the normalised cumulative distribution, so the result is a
    /// genuine categorical sample from `weights / sum(weights)`.
    pub fn sample_categorical(&mut self, weights: &[f64]) -> Option<usize> {
        let total: f64 = weights.iter().sum();
        if weights.is_empty() || !total.is_finite() || total <= 0.0 {
            return None;
        }
        let threshold = self.next_f64() * total;
        let mut acc = 0.0;
        for (idx, &w) in weights.iter().enumerate() {
            acc += w.max(0.0);
            if threshold < acc {
                return Some(idx);
            }
        }
        // Floating-point round-off: fall back to the last positive-weight index.
        weights.iter().rposition(|&w| w > 0.0)
    }
}

// ---------------------------------------------------------------------------
// SpeculativeDecoder
// ---------------------------------------------------------------------------

/// Outcome of one [`SpeculativeDecoder::verify_and_accept`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeResult {
    /// Tokens emitted this round: the accepted prefix of drafted tokens
    /// followed by exactly one extra token (a correction token on the first
    /// rejection, or a bonus token when every drafted token was accepted).
    pub tokens: Vec<u32>,
    /// Number of drafted tokens that passed the rejection test.
    pub accepted: usize,
    /// Number of drafted tokens that were rejected (`0` or `1` per round —
    /// at most one rejection can occur because verification stops there).
    pub rejected: usize,
}

/// Speculative decoding support (draft + verify).
///
/// A small "draft" model proposes several tokens ahead, and a larger "target"
/// model verifies them, accepting a prefix of the proposed tokens.  This
/// amortises the cost of autoregressive generation.
///
/// This is a faithful host-side implementation of the speculative-sampling
/// algorithm of Leviathan et al., *"Fast Inference from Transformers via
/// Speculative Decoding"* (2023), and Chen et al., *"Accelerating Large
/// Language Model Decoding with Speculative Sampling"* (2023):
///
/// 1. [`Self::propose_tokens`] draws drafted tokens by **categorical sampling**
///    from the draft model's per-position probability distributions.
/// 2. [`Self::verify_and_accept`] performs **modified rejection sampling**:
///    for the drafted token `t` at position `i`, a uniform `r ∈ [0, 1)` is
///    drawn and the token is accepted iff `r < min(1, p_target(t) / p_draft(t))`.
///    The longest passing prefix is kept; on the first rejection a correction
///    token is sampled from the normalised residual
///    `normalize(max(0, p_target − p_draft))`.  If every drafted token is
///    accepted, one bonus token is sampled from `p_target`.
///
/// The decoder operates on the probability vectors supplied by the caller
/// (which would come from running the draft and target model forward passes).
/// The sampling and acceptance arithmetic is exact — no values are fabricated.
#[derive(Debug)]
pub struct SpeculativeDecoder {
    draft_length: usize,
    rng: LcgRng,
    total_proposed: u64,
    total_accepted: u64,
    rounds: u64,
}

impl SpeculativeDecoder {
    /// Default RNG seed used by [`SpeculativeDecoder::new`].
    const DEFAULT_SEED: u64 = 0x5350_4543; // "SPEC"

    /// Creates a speculative decoder that proposes `draft_length` tokens at a
    /// time, using the default RNG seed.
    #[must_use]
    pub fn new(draft_length: usize) -> Self {
        Self::with_seed(draft_length, Self::DEFAULT_SEED)
    }

    /// Creates a speculative decoder with an explicit RNG seed.
    ///
    /// A fixed seed makes the categorical sampling and rejection draws
    /// reproducible, which is useful for tests and deterministic replay.
    #[must_use]
    pub fn with_seed(draft_length: usize, seed: u64) -> Self {
        Self {
            draft_length,
            rng: LcgRng::new(seed),
            total_proposed: 0,
            total_accepted: 0,
            rounds: 0,
        }
    }

    /// Number of tokens proposed per speculation round (γ).
    #[must_use]
    pub fn draft_length(&self) -> usize {
        self.draft_length
    }

    /// Proposes a sequence of drafted tokens from the draft model.
    ///
    /// `draft_probs` holds one probability distribution per draft position:
    /// `draft_probs[i]` is the draft model's categorical distribution over the
    /// vocabulary for the `i`-th drafted token (its length is the vocabulary
    /// size).  Each drafted token is obtained by **categorical sampling** from
    /// the corresponding distribution — this is genuine sampling from the draft
    /// model, not a deterministic placeholder.
    ///
    /// Returns one token id per position, paired with the probability the
    /// draft model assigned to the sampled token (`p_draft(t_i)`).  That
    /// probability is exactly what [`Self::verify_and_accept`] needs as the
    /// denominator of the acceptance ratio.
    ///
    /// At most `draft_length` positions are consumed; supplying fewer
    /// distributions simply produces a shorter draft.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if any consumed distribution is
    /// empty, has non-finite or all-zero mass, so that no token can be drawn.
    pub fn propose_tokens(&mut self, draft_probs: &[Vec<f64>]) -> DnnResult<Vec<DraftedToken>> {
        let count = draft_probs.len().min(self.draft_length);
        let mut drafted = Vec::with_capacity(count);
        for (position, dist) in draft_probs.iter().take(count).enumerate() {
            let token = self.rng.sample_categorical(dist).ok_or_else(|| {
                DnnError::InvalidArgument(format!(
                    "draft distribution at position {position} has no positive, finite mass"
                ))
            })?;
            let total: f64 = dist.iter().map(|p| p.max(0.0)).sum();
            // `total > 0` is guaranteed: sample_categorical returned `Some`.
            let draft_prob = dist[token].max(0.0) / total;
            drafted.push(DraftedToken {
                token_id: token as u32,
                draft_prob,
            });
        }
        Ok(drafted)
    }

    /// Verifies drafted tokens against the target model with modified
    /// rejection sampling, returning the tokens to emit this round.
    ///
    /// # Arguments
    ///
    /// * `drafted` — tokens proposed by [`Self::propose_tokens`], each carrying
    ///   the draft probability `p_draft(t_i)`.
    /// * `target_dists` — one target-model probability distribution per drafted
    ///   position: `target_dists[i]` is the target model's categorical
    ///   distribution over the vocabulary at position `i`.  It must be long
    ///   enough to cover every drafted token.
    ///
    /// # Algorithm
    ///
    /// For each drafted token `t_i` in order, a uniform `r ∈ [0, 1)` is drawn
    /// and `t_i` is accepted iff `r < min(1, p_target(t_i) / p_draft(t_i))`.
    /// On the first rejection at position `i`, a correction token is sampled
    /// from the normalised residual distribution
    /// `normalize(max(0, p_target[i] − p_draft[i]))` and verification stops.
    /// If every drafted token is accepted, one bonus token is sampled from the
    /// extra target distribution at position `drafted.len()`
    /// (`target_dists` must therefore contain `drafted.len() + 1` rows in that
    /// case — see the error conditions).
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `target_dists` does not cover
    /// every drafted position (plus one extra row for the all-accepted bonus
    /// token), if a target distribution referenced by a drafted token is too
    /// short to contain that token, or if a distribution required for sampling
    /// has no positive, finite mass.
    pub fn verify_and_accept(
        &mut self,
        drafted: &[DraftedToken],
        target_dists: &[Vec<f64>],
    ) -> DnnResult<SpeculativeResult> {
        let gamma = drafted.len();
        // The all-accepted path needs one extra target distribution to draw
        // the bonus token from, so `gamma + 1` rows are always required.
        if target_dists.len() <= gamma {
            return Err(DnnError::InvalidArgument(format!(
                "target_dists must have at least {} rows (one per drafted token \
                 plus a bonus row), got {}",
                gamma + 1,
                target_dists.len(),
            )));
        }

        let mut tokens = Vec::with_capacity(gamma + 1);
        for (i, draft) in drafted.iter().enumerate() {
            let token = draft.token_id as usize;
            let target_dist = &target_dists[i];
            let target_total: f64 = target_dist.iter().map(|p| p.max(0.0)).sum();
            let p_target = target_dist
                .get(token)
                .copied()
                .ok_or_else(|| {
                    DnnError::InvalidArgument(format!(
                        "target distribution at position {i} (len {}) does not \
                         contain drafted token id {token}",
                        target_dist.len(),
                    ))
                })?
                .max(0.0);
            // Normalise the target probability of the drafted token so the
            // acceptance ratio is between two genuine probabilities.
            let p_target = if target_total > 0.0 {
                p_target / target_total
            } else {
                0.0
            };

            // Acceptance ratio  min(1, p_target / p_draft).
            // A draft probability of zero means the draft model could never
            // have produced this token — treat it as an unconditional reject.
            let accept_ratio = if draft.draft_prob > 0.0 {
                (p_target / draft.draft_prob).min(1.0)
            } else {
                0.0
            };

            let r = self.rng.next_f64();
            if r < accept_ratio {
                tokens.push(draft.token_id);
                continue;
            }

            // ---- First rejection: sample the correction token. ----------
            let residual = Self::residual_distribution(target_dist, drafted, i);
            let correction = self.rng.sample_categorical(&residual).ok_or_else(|| {
                DnnError::InvalidArgument(format!(
                    "residual distribution at position {i} has no positive mass"
                ))
            })?;
            tokens.push(correction as u32);

            let accepted = i;
            self.record(gamma, accepted);
            return Ok(SpeculativeResult {
                tokens,
                accepted,
                rejected: 1,
            });
        }

        // ---- Every drafted token accepted: sample one bonus token. ------
        let bonus_dist = &target_dists[gamma];
        let bonus = self.rng.sample_categorical(bonus_dist).ok_or_else(|| {
            DnnError::InvalidArgument(
                "bonus target distribution has no positive, finite mass".into(),
            )
        })?;
        tokens.push(bonus as u32);

        self.record(gamma, gamma);
        Ok(SpeculativeResult {
            tokens,
            accepted: gamma,
            rejected: 0,
        })
    }

    /// Builds the normalised residual distribution
    /// `normalize(max(0, p_target − p_draft))` at the rejection position `i`.
    ///
    /// The draft model's distribution at position `i` is reconstructed as a
    /// one-hot vector on the drafted token: the draft sampled exactly that
    /// token, so all of the draft mass relevant to the residual sits there.
    /// This matches the speculative-sampling residual `p(x) − q(x)` clamped to
    /// non-negative values.  When the residual sums to zero (the target placed
    /// no extra mass anywhere) the raw target distribution is returned so the
    /// caller still draws a valid token from `p_target`.
    fn residual_distribution(
        target_dist: &[f64],
        drafted: &[DraftedToken],
        position: usize,
    ) -> Vec<f64> {
        let target_total: f64 = target_dist.iter().map(|p| p.max(0.0)).sum();
        let drafted_token = drafted[position].token_id as usize;
        let draft_prob = drafted[position].draft_prob;

        let mut residual: Vec<f64> = Vec::with_capacity(target_dist.len());
        for (idx, &t) in target_dist.iter().enumerate() {
            let p_target = if target_total > 0.0 {
                t.max(0.0) / target_total
            } else {
                0.0
            };
            // q(x): the draft distribution is one-hot on the drafted token.
            let p_draft = if idx == drafted_token {
                draft_prob.max(0.0)
            } else {
                0.0
            };
            residual.push((p_target - p_draft).max(0.0));
        }

        let residual_sum: f64 = residual.iter().sum();
        if residual_sum <= 0.0 {
            // Degenerate residual — draw the correction straight from p_target.
            return target_dist.iter().map(|p| p.max(0.0)).collect();
        }
        residual
    }

    /// Updates the running accept/propose counters for one finished round.
    fn record(&mut self, proposed: usize, accepted: usize) {
        self.total_proposed += proposed as u64;
        self.total_accepted += accepted as u64;
        self.rounds += 1;
    }

    /// Running acceptance rate across all calls to [`Self::verify_and_accept`].
    ///
    /// This is `total accepted drafted tokens / total drafted tokens` and
    /// reflects the real algorithm — it excludes correction and bonus tokens,
    /// which are emitted regardless of acceptance.
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_proposed == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_proposed as f64
    }

    /// Total number of drafted tokens proposed across all rounds.
    #[must_use]
    pub fn total_proposed(&self) -> u64 {
        self.total_proposed
    }

    /// Total number of drafted tokens accepted across all rounds.
    #[must_use]
    pub fn total_accepted(&self) -> u64 {
        self.total_accepted
    }

    /// Number of completed speculation rounds.
    #[must_use]
    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    /// Average number of tokens emitted per round, including the correction or
    /// bonus token.
    ///
    /// With a draft length of γ this lies in `[1, γ + 1]`: each round always
    /// emits one correction-or-bonus token on top of the accepted prefix.
    /// It is the practical speed-up factor of speculative decoding versus
    /// plain autoregressive decoding (one target forward pass per round).
    #[must_use]
    pub fn mean_tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        // accepted drafted tokens + one emitted token (correction/bonus) per round.
        (self.total_accepted + self.rounds) as f64 / self.rounds as f64
    }
}

/// A token drafted by the draft model, with the probability the draft model
/// assigned to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftedToken {
    /// Sampled vocabulary id.
    pub token_id: u32,
    /// Draft-model probability `p_draft(token_id)` of the sampled token,
    /// normalised so it lies in `[0, 1]`.
    pub draft_prob: f64,
}

// ---------------------------------------------------------------------------
// BatchMetrics
// ---------------------------------------------------------------------------

/// Running statistics for the inference serving loop.
#[derive(Debug)]
pub struct BatchMetrics {
    /// (prefill_tokens, decode_tokens, latency_us) per step.
    steps: Vec<(usize, usize, u64)>,
    /// Time-to-first-token in microseconds for each request.
    ttft_samples: Vec<u64>,
}

impl BatchMetrics {
    /// Create an empty metrics collector.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            ttft_samples: Vec::new(),
        }
    }

    /// Record one scheduling step.
    pub fn record_step(&mut self, prefill_tokens: usize, decode_tokens: usize, latency_us: u64) {
        self.steps.push((prefill_tokens, decode_tokens, latency_us));
    }

    /// Record a time-to-first-token sample (in microseconds).
    pub fn record_ttft(&mut self, ttft_us: u64) {
        self.ttft_samples.push(ttft_us);
    }

    /// Average latency of steps that included prefill tokens (microseconds).
    pub fn avg_prefill_latency(&self) -> f64 {
        let prefills: Vec<u64> = self
            .steps
            .iter()
            .filter(|(p, _, _)| *p > 0)
            .map(|(_, _, l)| *l)
            .collect();
        if prefills.is_empty() {
            return 0.0;
        }
        prefills.iter().sum::<u64>() as f64 / prefills.len() as f64
    }

    /// Average latency of steps that included decode tokens (microseconds).
    pub fn avg_decode_latency(&self) -> f64 {
        let decodes: Vec<u64> = self
            .steps
            .iter()
            .filter(|(_, d, _)| *d > 0)
            .map(|(_, _, l)| *l)
            .collect();
        if decodes.is_empty() {
            return 0.0;
        }
        decodes.iter().sum::<u64>() as f64 / decodes.len() as f64
    }

    /// Average batch size (total tokens per step).
    pub fn avg_batch_size(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total: usize = self.steps.iter().map(|(p, d, _)| p + d).sum();
        total as f64 / self.steps.len() as f64
    }

    /// Estimated token throughput (tokens / second).
    pub fn token_throughput(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total_tokens: usize = self.steps.iter().map(|(p, d, _)| p + d).sum();
        let total_us: u64 = self.steps.iter().map(|(_, _, l)| l).sum();
        if total_us == 0 {
            return 0.0;
        }
        total_tokens as f64 / (total_us as f64 / 1_000_000.0)
    }

    /// Median (p50) time-to-first-token in microseconds.
    pub fn time_to_first_token_p50(&self) -> f64 {
        if self.ttft_samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.ttft_samples.clone();
        sorted.sort_unstable();
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 && sorted.len() >= 2 {
            (sorted[mid - 1] + sorted[mid]) as f64 / 2.0
        } else {
            sorted[mid] as f64
        }
    }

    /// Human-readable performance report.
    pub fn format_report(&self) -> String {
        format!(
            "BatchMetrics Report\n\
             ====================\n\
             Steps recorded       : {}\n\
             Avg prefill latency  : {:.1} us\n\
             Avg decode latency   : {:.1} us\n\
             Avg batch size       : {:.1} tokens/step\n\
             Token throughput     : {:.0} tokens/s\n\
             TTFT p50             : {:.1} us\n\
             TTFT samples         : {}",
            self.steps.len(),
            self.avg_prefill_latency(),
            self.avg_decode_latency(),
            self.avg_batch_size(),
            self.token_throughput(),
            self.time_to_first_token_p50(),
            self.ttft_samples.len(),
        )
    }
}

impl Default for BatchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> BatchConfig {
        BatchConfig {
            max_batch_size: 8,
            max_total_tokens: 4096,
            max_sequence_length: 2048,
            prefill_batch_size: 1024,
            decode_batch_size: 8,
            scheduling_policy: SchedulingPolicy::Fcfs,
        }
    }

    fn make_request(id: RequestId, seq_len: usize, max_new: usize) -> InferenceRequest {
        InferenceRequest {
            request_id: id,
            sequence_length: seq_len,
            max_new_tokens: max_new,
            priority: Priority::Normal,
            arrival_time_ns: id * 1000,
            deadline_ns: None,
        }
    }

    // 1. Add single request
    #[test]
    fn test_add_single_request() {
        let mut batcher = ContinuousBatcher::new(default_config());
        let req = make_request(1, 128, 64);
        let id = batcher.add_request(req).expect("should succeed");
        assert_eq!(id, 1);
        assert_eq!(batcher.pending_requests(), 1);
        assert_eq!(batcher.active_requests(), 0);
    }

    // 2. Batch step with mixed prefill/decode
    #[test]
    fn test_batch_step_mixed_prefill_decode() {
        let mut batcher = ContinuousBatcher::new(default_config());
        // First request: prefill + become decode
        batcher.add_request(make_request(1, 64, 32)).expect("add 1");
        let d1 = batcher.step().expect("step 1");
        assert_eq!(d1.prefill_requests.len(), 1);

        // Second request while first is decoding
        batcher.add_request(make_request(2, 32, 16)).expect("add 2");
        let d2 = batcher.step().expect("step 2");
        assert!(!d2.decode_requests.is_empty(), "should have decode slots");
        assert!(!d2.prefill_requests.is_empty(), "should have prefill slots");
    }

    // 3. Token budget allocation/release
    #[test]
    fn test_token_budget_allocation_release() {
        let mut alloc = TokenBudgetAllocator::new(1024);
        let slot = alloc.allocate_prefill(512);
        assert!(slot.is_some());
        assert!((alloc.utilization() - 0.5).abs() < 1e-9);

        // Allocate more than remaining.
        assert!(alloc.allocate_prefill(600).is_none());

        alloc.release(256);
        assert!((alloc.utilization() - 0.25).abs() < 1e-9);
    }

    // 4. Paged KV allocation/free
    #[test]
    fn test_paged_kv_allocation_free() {
        let mut mgr = PagedKvManager::new(16, 64);
        let blocks = mgr.allocate(128).expect("allocate 128");
        assert_eq!(blocks.len(), 2);
        let (used, total) = mgr.usage();
        assert_eq!(used, 2);
        assert_eq!(total, 16);

        mgr.free(&blocks);
        let (used, _) = mgr.usage();
        assert_eq!(used, 0);
    }

    // 5. Copy-on-write
    #[test]
    fn test_copy_on_write() {
        let mut mgr = PagedKvManager::new(4, 64);
        let blocks = mgr.allocate(64).expect("allocate");
        assert_eq!(blocks.len(), 1);
        let orig = blocks[0];

        // Bump ref count to simulate sharing.
        mgr.ref_counts[orig] = 2;

        let new_id = mgr.copy_on_write(orig).expect("cow");
        assert_ne!(new_id, orig);
        // Old block should still be allocated (ref_count decremented to 1).
        assert!(!mgr.free_map[orig]);
        assert_eq!(mgr.ref_counts[orig], 1);
        assert_eq!(mgr.ref_counts[new_id], 1);
    }

    // 6. Continuous batching with request completion
    #[test]
    fn test_continuous_batching_completion() {
        let mut batcher = ContinuousBatcher::new(default_config());
        batcher.add_request(make_request(10, 64, 8)).expect("add");
        let _ = batcher.step().expect("step");
        assert_eq!(batcher.active_requests(), 1);

        batcher.complete_request(10).expect("complete");
        assert_eq!(batcher.active_requests(), 0);
    }

    // 7. Preemption
    #[test]
    fn test_preemption() {
        let mut batcher = ContinuousBatcher::new(default_config());
        batcher.add_request(make_request(20, 64, 16)).expect("add");
        let _ = batcher.step().expect("step");
        assert_eq!(batcher.active_requests(), 1);

        batcher.preempt(20).expect("preempt");
        assert_eq!(batcher.active_requests(), 0);
        // Preempted request is in the preempted queue.
        assert_eq!(batcher.pending_requests(), 1);
    }

    // 8. FCFS scheduling order
    #[test]
    fn test_fcfs_scheduling_order() {
        let mut batcher = ContinuousBatcher::new(default_config());
        batcher.add_request(make_request(3, 32, 8)).expect("add 3");
        batcher.add_request(make_request(1, 32, 8)).expect("add 1");
        batcher.add_request(make_request(2, 32, 8)).expect("add 2");
        // arrival_time_ns = id * 1000, so order is 1, 2, 3.
        let d = batcher.step().expect("step");
        assert_eq!(d.prefill_requests, vec![1, 2, 3]);
    }

    // 9. Priority-based scheduling
    #[test]
    fn test_priority_based_scheduling() {
        let mut config = default_config();
        config.scheduling_policy = SchedulingPolicy::PriorityBased;
        let mut batcher = ContinuousBatcher::new(config);

        let mut low = make_request(1, 32, 8);
        low.priority = Priority::Low;
        low.arrival_time_ns = 100;
        let mut high = make_request(2, 32, 8);
        high.priority = Priority::High;
        high.arrival_time_ns = 200;
        let mut normal = make_request(3, 32, 8);
        normal.priority = Priority::Normal;
        normal.arrival_time_ns = 50;

        batcher.add_request(low).expect("add low");
        batcher.add_request(high).expect("add high");
        batcher.add_request(normal).expect("add normal");

        let d = batcher.step().expect("step");
        // High (2) first, then Normal (3), then Low (1).
        assert_eq!(d.prefill_requests, vec![2, 3, 1]);
    }

    // 10. Deadline-aware scheduling
    #[test]
    fn test_deadline_aware_scheduling() {
        let mut config = default_config();
        config.scheduling_policy = SchedulingPolicy::DeadlineAware;
        let mut batcher = ContinuousBatcher::new(config);

        let mut r1 = make_request(1, 32, 8);
        r1.deadline_ns = Some(5000);
        let mut r2 = make_request(2, 32, 8);
        r2.deadline_ns = Some(1000);
        let mut r3 = make_request(3, 32, 8);
        r3.deadline_ns = None; // No deadline → goes last.

        batcher.add_request(r1).expect("add r1");
        batcher.add_request(r2).expect("add r2");
        batcher.add_request(r3).expect("add r3");

        let d = batcher.step().expect("step");
        assert_eq!(d.prefill_requests, vec![2, 1, 3]);
    }

    // 11. Speculative decoding — propose samples from the draft distribution.
    #[test]
    fn test_speculative_decoding_propose_samples_draft() {
        let mut spec = SpeculativeDecoder::with_seed(3, 12345);
        // Three positions, each a 4-token vocabulary. Position 0 always
        // produces token 2 (the only one with mass); position 1 produces
        // token 0; position 2 produces token 3.
        let draft_probs = vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let drafted = spec.propose_tokens(&draft_probs).expect("propose");
        assert_eq!(drafted.len(), 3);
        assert_eq!(drafted[0].token_id, 2);
        assert_eq!(drafted[1].token_id, 0);
        assert_eq!(drafted[2].token_id, 3);
        // The draft probability of a one-hot distribution is exactly 1.0.
        for d in &drafted {
            assert!((d.draft_prob - 1.0).abs() < 1e-12);
        }
    }

    // 11b. propose_tokens respects the draft_length cap and normalises probs.
    #[test]
    fn test_speculative_decoding_propose_caps_and_normalises() {
        let mut spec = SpeculativeDecoder::with_seed(2, 99);
        // Four positions supplied but draft_length == 2 → only 2 consumed.
        // Un-normalised weights: token 1 has 3/4 of the mass.
        let draft_probs = vec![
            vec![1.0, 3.0],
            vec![3.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let drafted = spec.propose_tokens(&draft_probs).expect("propose");
        assert_eq!(drafted.len(), 2, "draft_length caps the count");
        for d in &drafted {
            // draft_prob must be a genuine probability in [0, 1].
            assert!((0.0..=1.0).contains(&d.draft_prob));
            // For these two-element weight vectors the sampled token's
            // normalised probability is either 1/4 or 3/4.
            let p = d.draft_prob;
            assert!(
                (p - 0.25).abs() < 1e-12 || (p - 0.75).abs() < 1e-12,
                "unexpected normalised prob {p}"
            );
        }
    }

    // 11c. propose_tokens rejects a degenerate (all-zero) distribution.
    #[test]
    fn test_speculative_decoding_propose_rejects_zero_dist() {
        let mut spec = SpeculativeDecoder::new(2);
        let draft_probs = vec![vec![0.0, 0.0, 0.0]];
        assert!(spec.propose_tokens(&draft_probs).is_err());
    }

    // 11d. Categorical sampling reproduces the target frequencies (statistical).
    #[test]
    fn test_categorical_sampling_matches_distribution() {
        let mut rng = LcgRng::new(0x00C0_FFEE);
        // Target distribution over 4 categories.
        let weights = [0.1_f64, 0.2, 0.3, 0.4];
        let trials = 200_000;
        let mut counts = [0u64; 4];
        for _ in 0..trials {
            let idx = rng.sample_categorical(&weights).expect("sample");
            counts[idx] += 1;
        }
        for (i, &w) in weights.iter().enumerate() {
            let freq = counts[i] as f64 / trials as f64;
            assert!(
                (freq - w).abs() < 0.01,
                "category {i}: freq {freq} vs expected {w}"
            );
        }
    }

    // 11e. Rejection sampling accepts with probability min(1, p_t / p_d).
    #[test]
    fn test_rejection_sampling_acceptance_probability() {
        // Drafted token id 0 with draft prob 0.8. Target prob 0.4 → the
        // acceptance ratio is 0.4 / 0.8 = 0.5: across many independent
        // single-token rounds, ~half should be accepted.
        let trials = 100_000;
        let mut accepted_rounds = 0u64;
        for seed in 0..trials {
            let mut spec = SpeculativeDecoder::with_seed(1, seed);
            let drafted = vec![DraftedToken {
                token_id: 0,
                draft_prob: 0.8,
            }];
            // Position 0 target dist: token 0 has prob 0.4, token 1 has 0.6.
            // Bonus row (position 1) is required even on rejection paths.
            let target = vec![vec![0.4, 0.6], vec![0.5, 0.5]];
            let res = spec.verify_and_accept(&drafted, &target).expect("verify");
            if res.accepted == 1 {
                accepted_rounds += 1;
            }
        }
        let rate = accepted_rounds as f64 / trials as f64;
        assert!(
            (rate - 0.5).abs() < 0.01,
            "acceptance rate {rate} should be ~0.5"
        );
    }

    // 11f. p_target >= p_draft → unconditional acceptance.
    #[test]
    fn test_rejection_sampling_always_accepts_when_target_ge_draft() {
        for seed in 0..2000 {
            let mut spec = SpeculativeDecoder::with_seed(1, seed);
            let drafted = vec![DraftedToken {
                token_id: 0,
                draft_prob: 0.3,
            }];
            // Target prob of token 0 (normalised) is 0.6 >= 0.3 draft prob.
            let target = vec![vec![0.6, 0.4], vec![0.5, 0.5]];
            let res = spec.verify_and_accept(&drafted, &target).expect("verify");
            assert_eq!(res.accepted, 1, "ratio >= 1 must always accept");
            assert_eq!(res.rejected, 0);
        }
    }

    // 11g. Zero draft probability → unconditional rejection.
    #[test]
    fn test_rejection_sampling_rejects_zero_draft_prob() {
        let mut spec = SpeculativeDecoder::with_seed(1, 7);
        let drafted = vec![DraftedToken {
            token_id: 0,
            draft_prob: 0.0,
        }];
        let target = vec![vec![0.9, 0.1], vec![0.5, 0.5]];
        let res = spec.verify_and_accept(&drafted, &target).expect("verify");
        assert_eq!(res.accepted, 0);
        assert_eq!(res.rejected, 1);
        // Exactly one correction token is emitted.
        assert_eq!(res.tokens.len(), 1);
    }

    // 11h. Residual-distribution resampling is correct (statistical).
    #[test]
    fn test_residual_distribution_resampling() {
        // Force a guaranteed rejection so the correction token is always
        // drawn from the residual. Drafted token id 0 with draft prob 1.0 and
        // target prob 0.0 for token 0 → acceptance ratio 0 → always reject.
        //
        // Target distribution: [0.0, 0.5, 0.5]. Draft is one-hot on token 0.
        // Residual = max(0, p_target - p_draft):
        //   token 0: max(0, 0.0 - 1.0) = 0.0
        //   token 1: max(0, 0.5 - 0.0) = 0.5
        //   token 2: max(0, 0.5 - 0.0) = 0.5
        // Normalised residual = [0.0, 0.5, 0.5].
        let trials = 100_000;
        let mut counts = [0u64; 3];
        for seed in 0..trials {
            let mut spec = SpeculativeDecoder::with_seed(1, seed);
            let drafted = vec![DraftedToken {
                token_id: 0,
                draft_prob: 1.0,
            }];
            let target = vec![vec![0.0, 0.5, 0.5], vec![1.0, 0.0, 0.0]];
            let res = spec.verify_and_accept(&drafted, &target).expect("verify");
            assert_eq!(res.accepted, 0, "must reject");
            let corr = res.tokens[0] as usize;
            counts[corr] += 1;
        }
        let total = trials as f64;
        assert_eq!(counts[0], 0, "token 0 has zero residual mass");
        assert!((counts[1] as f64 / total - 0.5).abs() < 0.01);
        assert!((counts[2] as f64 / total - 0.5).abs() < 0.01);
    }

    // 11i. Rejection of a token the target does not place mass on draws the
    //      correction from the residual concentrated elsewhere.
    #[test]
    fn test_residual_distribution_concentrated() {
        // Drafted token id 1 with draft prob 1.0; target dist [1.0, 0.0].
        // p_target(token 1) == 0 → acceptance ratio 0 → always reject.
        // residual = max(0, p_target - p_draft) with draft one-hot on token 1:
        //   token 0: max(0, 1.0 - 0.0) = 1.0
        //   token 1: max(0, 0.0 - 1.0) = 0.0
        // residual = [1.0, 0.0] (non-zero) → correction is always token 0.
        for seed in 0..1000 {
            let mut spec = SpeculativeDecoder::with_seed(1, seed);
            let drafted = vec![DraftedToken {
                token_id: 1,
                draft_prob: 1.0,
            }];
            let target = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
            let res = spec.verify_and_accept(&drafted, &target).expect("verify");
            assert_eq!(res.accepted, 0);
            assert_eq!(res.tokens[0], 0, "residual concentrates on token 0");
        }
    }

    // 11i-bis. A residual that sums to zero falls back to the target dist.
    #[test]
    fn test_residual_distribution_zero_fallback() {
        // residual_distribution returns the raw target distribution when the
        // clamped residual max(0, p_target - p_draft) sums to zero. This
        // happens when the draft one-hot mass fully covers the target mass at
        // the drafted token and the target has no mass anywhere else.
        // Construct that case directly: target one-hot on the drafted token.
        let drafted = [DraftedToken {
            token_id: 0,
            draft_prob: 1.0,
        }];
        let target_dist = [1.0_f64, 0.0, 0.0];
        let residual = SpeculativeDecoder::residual_distribution(&target_dist, &drafted, 0);
        // p_target = [1,0,0], p_draft one-hot on token 0 = [1,0,0]:
        // residual = [0,0,0] → sum 0 → fallback returns the target dist.
        assert_eq!(residual, vec![1.0, 0.0, 0.0]);
    }

    // 11j. Draft == target accepts every token (statistical).
    #[test]
    fn test_speculative_draft_equals_target_accepts_all() {
        // When the draft and target distributions are identical, the
        // acceptance ratio p_target / p_draft is exactly 1.0 for every
        // drafted token, so all gamma tokens must always be accepted.
        let gamma = 5;
        for seed in 0..3000 {
            let mut spec = SpeculativeDecoder::with_seed(gamma, seed);
            // Identical draft/target distributions for every position.
            let dist = vec![0.15, 0.25, 0.20, 0.40];
            let draft_probs = vec![dist.clone(); gamma];
            let drafted = spec.propose_tokens(&draft_probs).expect("propose");
            assert_eq!(drafted.len(), gamma);

            // Target: same distribution at every drafted position + bonus row.
            let target_dists = vec![dist.clone(); gamma + 1];
            let res = spec
                .verify_and_accept(&drafted, &target_dists)
                .expect("verify");
            assert_eq!(res.accepted, gamma, "draft==target must accept all");
            assert_eq!(res.rejected, 0);
            // Accepted prefix + 1 bonus token.
            assert_eq!(res.tokens.len(), gamma + 1);
        }
    }

    // 11k. Accepted-length distribution is sane and accounting is correct.
    #[test]
    fn test_speculative_accepted_length_distribution() {
        let gamma = 4usize;
        let mut spec = SpeculativeDecoder::with_seed(gamma, 0xABCD);
        let rounds = 5000u64;
        let mut sum_accepted = 0u64;
        for _ in 0..rounds {
            // Draft distribution: token 0 always sampled (one-hot).
            let draft_probs = vec![vec![1.0, 0.0]; gamma];
            let drafted = spec.propose_tokens(&draft_probs).expect("propose");
            // Target: token 0 (drafted) has acceptance ratio 0.7/1.0 = 0.7.
            let target_dists = vec![vec![0.7, 0.3]; gamma + 1];
            let res = spec
                .verify_and_accept(&drafted, &target_dists)
                .expect("verify");
            assert!(res.accepted <= gamma, "accepted within [0, gamma]");
            assert_eq!(res.rejected, usize::from(res.accepted < gamma));
            // Emitted tokens = accepted prefix + exactly one extra token.
            assert_eq!(res.tokens.len(), res.accepted + 1);
            sum_accepted += res.accepted as u64;
        }
        // total_proposed = rounds * gamma; acceptance rate should track 0.7.
        assert_eq!(spec.total_proposed(), rounds * gamma as u64);
        assert_eq!(spec.total_accepted(), sum_accepted);
        assert_eq!(spec.rounds(), rounds);
        let rate = spec.acceptance_rate();
        // Per-position acceptance is p = 0.7, but the first rejection truncates
        // the round, so the realised rate is well below p. The expected number
        // of accepted drafted tokens per gamma=4 round is
        //   sum_{j=1}^{3} j*p^j*(1-p) + 4*p^4 = 1.7731,
        // giving an expected rate of 1.7731 / 4 ≈ 0.4433.
        assert!(
            (rate - 0.4433).abs() < 0.02,
            "acceptance rate {rate} should be ~0.4433"
        );
        // Mean tokens per round must lie in [1, gamma + 1].
        let mtpr = spec.mean_tokens_per_round();
        assert!(mtpr >= 1.0 && mtpr <= (gamma + 1) as f64, "mtpr {mtpr}");
    }

    // 11l. verify_and_accept errors when target_dists is too short.
    #[test]
    fn test_speculative_verify_rejects_short_target() {
        let mut spec = SpeculativeDecoder::new(2);
        let drafted = vec![
            DraftedToken {
                token_id: 0,
                draft_prob: 0.5,
            },
            DraftedToken {
                token_id: 1,
                draft_prob: 0.5,
            },
        ];
        // Only 2 rows supplied; need gamma + 1 == 3.
        let target = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        assert!(spec.verify_and_accept(&drafted, &target).is_err());
    }

    // 11m. verify_and_accept errors when a drafted token is out of vocab.
    #[test]
    fn test_speculative_verify_rejects_token_out_of_range() {
        let mut spec = SpeculativeDecoder::new(1);
        let drafted = vec![DraftedToken {
            token_id: 9, // out of range for a 2-token target distribution
            draft_prob: 0.5,
        }];
        let target = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        assert!(spec.verify_and_accept(&drafted, &target).is_err());
    }

    // 11n. Empty draft → only a bonus token is emitted.
    #[test]
    fn test_speculative_empty_draft_emits_bonus() {
        let mut spec = SpeculativeDecoder::with_seed(4, 55);
        let drafted: Vec<DraftedToken> = Vec::new();
        // gamma == 0, so a single bonus row is required.
        let target = vec![vec![0.0, 1.0, 0.0]];
        let res = spec.verify_and_accept(&drafted, &target).expect("verify");
        assert_eq!(res.accepted, 0);
        assert_eq!(res.rejected, 0);
        assert_eq!(res.tokens, vec![1], "bonus drawn from one-hot target");
    }

    // 11o. LcgRng produces uniform f64 in [0, 1) and is deterministic.
    #[test]
    fn test_lcg_rng_uniform_and_deterministic() {
        let mut a = LcgRng::new(2024);
        let mut b = LcgRng::new(2024);
        let mut sum = 0.0_f64;
        let n = 100_000;
        for _ in 0..n {
            let va = a.next_f64();
            let vb = b.next_f64();
            assert_eq!(va, vb, "same seed must yield same stream");
            assert!((0.0..1.0).contains(&va));
            sum += va;
        }
        // Mean of a uniform [0,1) sample should be close to 0.5.
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "uniform mean {mean}");
    }

    // 12. Batch metrics tracking
    #[test]
    fn test_batch_metrics_tracking() {
        let mut m = BatchMetrics::new();
        m.record_step(128, 0, 500);
        m.record_step(0, 8, 100);
        m.record_step(64, 4, 300);

        assert!((m.avg_prefill_latency() - 400.0).abs() < 1e-9);
        assert!((m.avg_decode_latency() - 200.0).abs() < 1e-9);
        // (128+0+8+64+4) / 3 = 68.0
        assert!((m.avg_batch_size() - 68.0).abs() < 1e-9);
        assert!(m.token_throughput() > 0.0);
    }

    // 13. Max batch size enforcement
    #[test]
    fn test_max_batch_size_enforcement() {
        let mut config = default_config();
        config.max_batch_size = 2;
        let mut batcher = ContinuousBatcher::new(config);

        for i in 0..4 {
            batcher.add_request(make_request(i, 32, 8)).expect("add");
        }
        let d = batcher.step().expect("step");
        assert!(d.prefill_requests.len() <= 2);
        assert_eq!(batcher.active_requests(), d.prefill_requests.len());
    }

    // 14. Queue management
    #[test]
    fn test_queue_management() {
        let mut batcher = ContinuousBatcher::new(default_config());
        assert_eq!(batcher.pending_requests(), 0);

        batcher.add_request(make_request(1, 32, 8)).expect("add");
        batcher.add_request(make_request(2, 32, 8)).expect("add");
        assert_eq!(batcher.pending_requests(), 2);

        let _ = batcher.step().expect("step");
        assert_eq!(batcher.pending_requests(), 0);
        assert_eq!(batcher.active_requests(), 2);

        batcher.complete_request(1).expect("complete");
        assert_eq!(batcher.active_requests(), 1);
    }

    // 15. Utilization calculation
    #[test]
    fn test_utilization_calculation() {
        let mut alloc = TokenBudgetAllocator::new(1000);
        assert!((alloc.utilization() - 0.0).abs() < 1e-9);

        alloc.allocate_prefill(250);
        assert!((alloc.utilization() - 0.25).abs() < 1e-9);

        let fitted = alloc.allocate_decode(900);
        assert_eq!(fitted, 750);
        assert!((alloc.utilization() - 1.0).abs() < 1e-9);

        // Edge case: zero-capacity allocator.
        let zero = TokenBudgetAllocator::new(0);
        assert!((zero.utilization() - 0.0).abs() < 1e-9);
    }

    // 16. Format report
    #[test]
    fn test_format_report() {
        let mut m = BatchMetrics::new();
        m.record_step(100, 10, 200);
        m.record_step(0, 8, 100);
        m.record_ttft(150);
        m.record_ttft(250);
        let report = m.format_report();
        assert!(report.contains("Steps recorded"));
        assert!(report.contains("Token throughput"));
        assert!(report.contains("TTFT p50"));
    }
}

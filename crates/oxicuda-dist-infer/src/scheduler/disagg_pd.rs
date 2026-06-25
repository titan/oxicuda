//! Disaggregated prefill / decode scheduler (DistServe, Zhong 2024).
//!
//! Prefill (processing the whole prompt) and decode (generating one token at a
//! time) have opposite resource profiles: prefill is compute-bound and bursty;
//! decode is memory-bandwidth-bound and long-lived. Co-locating them on the
//! same workers couples their latencies — a long prefill stalls every decode on
//! that GPU. **Disaggregation** runs two separate worker pools and *migrates the
//! KV cache* from a prefill worker to a decode worker after the prompt is
//! processed.
//!
//! This module is the scheduling logic for that split: it picks the
//! least-loaded prefill worker for an incoming request, records the KV-cache
//! [`PrefillHandoff`] to be transferred over the interconnect, and assigns the
//! decode worker that will own the sequence to completion. The *transfer itself*
//! (a cross-rank P2P copy) is hardware work handled by `BlockMigrator` /
//! `oxicuda-driver`; here we plan it and account for load.
//!
//! Exact oracles: every request flows prefill → handoff → decode exactly once;
//! per-pool load counts are conserved; the chosen worker is always a least-loaded
//! one.

use std::collections::HashMap;

use crate::error::{DistInferError, DistInferResult};
use crate::router::request::Request;

/// Which pool a worker belongs to / which phase a request is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdPhase {
    /// Processing the prompt (compute-bound).
    Prefill,
    /// Generating tokens (memory-bound).
    Decode,
}

/// A planned KV-cache hand-off from a prefill worker to a decode worker.
///
/// Carries everything `BlockMigrator` needs to move the prompt's KV blocks
/// across the interconnect. The migration *execution* requires real hardware;
/// this descriptor is the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillHandoff {
    /// Request being handed off.
    pub request_id: u64,
    /// Prefill worker that produced the KV cache.
    pub prefill_worker: usize,
    /// Decode worker that will own the sequence.
    pub decode_worker: usize,
    /// Number of prompt tokens (KV entries) to transfer.
    pub prompt_tokens: usize,
    /// Number of KV blocks to transfer.
    pub n_blocks: usize,
}

/// Cumulative scheduler statistics (goodput accounting).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PdStats {
    /// Requests routed to prefill.
    pub prefilled: u64,
    /// Hand-offs planned (prefill → decode).
    pub handoffs: u64,
    /// Requests that completed decode.
    pub completed: u64,
    /// Total KV blocks moved across the interconnect.
    pub blocks_migrated: u64,
}

/// Disaggregated prefill/decode scheduler.
#[derive(Debug)]
pub struct DisaggPdScheduler {
    block_size: usize,
    /// In-flight request count per prefill worker.
    prefill_load: Vec<usize>,
    /// In-flight (running) sequence count per decode worker.
    decode_load: Vec<usize>,
    /// request_id → its current phase + owning worker.
    location: HashMap<u64, (PdPhase, usize)>,
    stats: PdStats,
}

impl DisaggPdScheduler {
    /// Construct a scheduler with `n_prefill` prefill workers and `n_decode`
    /// decode workers, paged KV at `block_size` tokens/block.
    ///
    /// # Errors
    ///
    /// [`DistInferError::TooFewRanks`] if either pool is empty, or
    /// [`DistInferError::DimensionMismatch`] if `block_size == 0`.
    pub fn new(n_prefill: usize, n_decode: usize, block_size: usize) -> DistInferResult<Self> {
        if n_prefill == 0 || n_decode == 0 {
            return Err(DistInferError::TooFewRanks {
                needed: 1,
                world_size: n_prefill.min(n_decode),
            });
        }
        if block_size == 0 {
            return Err(DistInferError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        Ok(Self {
            block_size,
            prefill_load: vec![0; n_prefill],
            decode_load: vec![0; n_decode],
            location: HashMap::new(),
            stats: PdStats::default(),
        })
    }

    /// Number of prefill workers.
    #[must_use]
    pub fn n_prefill(&self) -> usize {
        self.prefill_load.len()
    }

    /// Number of decode workers.
    #[must_use]
    pub fn n_decode(&self) -> usize {
        self.decode_load.len()
    }

    /// Per-prefill-worker in-flight load.
    #[must_use]
    pub fn prefill_load(&self) -> &[usize] {
        &self.prefill_load
    }

    /// Per-decode-worker in-flight load.
    #[must_use]
    pub fn decode_load(&self) -> &[usize] {
        &self.decode_load
    }

    /// Cumulative statistics.
    #[must_use]
    pub fn stats(&self) -> &PdStats {
        &self.stats
    }

    /// Current phase + worker of a request, if tracked.
    #[must_use]
    pub fn locate(&self, request_id: u64) -> Option<(PdPhase, usize)> {
        self.location.get(&request_id).copied()
    }

    fn least_loaded(load: &[usize]) -> usize {
        load.iter()
            .enumerate()
            .min_by_key(|&(_, &l)| l)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Admit a request to the least-loaded prefill worker. Returns that worker.
    ///
    /// # Errors
    ///
    /// * [`DistInferError::EmptyTokenSequence`] if the prompt is empty.
    /// * [`DistInferError::Internal`] if the request is already in flight.
    pub fn schedule_prefill(&mut self, req: &Request) -> DistInferResult<usize> {
        if req.token_ids.is_empty() {
            return Err(DistInferError::EmptyTokenSequence);
        }
        if self.location.contains_key(&req.request_id) {
            return Err(DistInferError::Internal(
                "request already scheduled in the prefill/decode pipeline",
            ));
        }
        let worker = Self::least_loaded(&self.prefill_load);
        self.prefill_load[worker] += 1;
        self.location
            .insert(req.request_id, (PdPhase::Prefill, worker));
        self.stats.prefilled += 1;
        Ok(worker)
    }

    /// Complete prefill for `request_id` and plan the hand-off to the
    /// least-loaded decode worker. Frees the prefill worker's slot, charges the
    /// decode worker, and returns the [`PrefillHandoff`] plan.
    ///
    /// `prompt_tokens` is the prompt length (drives the block count migrated).
    ///
    /// # Errors
    ///
    /// [`DistInferError::Internal`] if the request is not currently in the
    /// prefill phase.
    pub fn complete_prefill(
        &mut self,
        request_id: u64,
        prompt_tokens: usize,
    ) -> DistInferResult<PrefillHandoff> {
        let (phase, prefill_worker) =
            self.location
                .get(&request_id)
                .copied()
                .ok_or(DistInferError::Internal(
                    "complete_prefill for an unknown request",
                ))?;
        if phase != PdPhase::Prefill {
            return Err(DistInferError::Internal(
                "complete_prefill on a request not in the prefill phase",
            ));
        }
        // Release prefill slot.
        self.prefill_load[prefill_worker] = self.prefill_load[prefill_worker].saturating_sub(1);
        // Assign decode worker.
        let decode_worker = Self::least_loaded(&self.decode_load);
        self.decode_load[decode_worker] += 1;
        self.location
            .insert(request_id, (PdPhase::Decode, decode_worker));

        let n_blocks = prompt_tokens.div_ceil(self.block_size);
        self.stats.handoffs += 1;
        self.stats.blocks_migrated += n_blocks as u64;

        Ok(PrefillHandoff {
            request_id,
            prefill_worker,
            decode_worker,
            prompt_tokens,
            n_blocks,
        })
    }

    /// Retire a sequence that finished decoding, freeing its decode-worker slot.
    ///
    /// # Errors
    ///
    /// [`DistInferError::Internal`] if the request is not in the decode phase.
    pub fn complete_decode(&mut self, request_id: u64) -> DistInferResult<()> {
        let (phase, decode_worker) =
            self.location
                .get(&request_id)
                .copied()
                .ok_or(DistInferError::Internal(
                    "complete_decode for an unknown request",
                ))?;
        if phase != PdPhase::Decode {
            return Err(DistInferError::Internal(
                "complete_decode on a request not in the decode phase",
            ));
        }
        self.decode_load[decode_worker] = self.decode_load[decode_worker].saturating_sub(1);
        self.location.remove(&request_id);
        self.stats.completed += 1;
        Ok(())
    }

    /// Total in-flight requests across both pools.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.location.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn req(id: u64, prompt: usize) -> Request {
        Request {
            request_id: id,
            token_ids: vec![1u32; prompt],
            max_new_tokens: 16,
            priority: 0,
        }
    }

    #[test]
    fn prefill_picks_least_loaded() {
        let mut s = DisaggPdScheduler::new(3, 2, 16).expect("sched");
        let w0 = s.schedule_prefill(&req(1, 10)).expect("p1");
        let w1 = s.schedule_prefill(&req(2, 10)).expect("p2");
        let w2 = s.schedule_prefill(&req(3, 10)).expect("p3");
        // Three distinct least-loaded workers in round-robin-by-load order.
        assert_eq!((w0, w1, w2), (0, 1, 2));
        assert_eq!(s.prefill_load(), &[1, 1, 1]);
    }

    #[test]
    fn full_pipeline_conserves_load() {
        let mut s = DisaggPdScheduler::new(2, 2, 8).expect("sched");
        let pw = s.schedule_prefill(&req(1, 16)).expect("prefill");
        assert_eq!(s.prefill_load()[pw], 1);
        let h = s.complete_prefill(1, 16).expect("handoff");
        // Prefill slot freed, decode slot charged.
        assert_eq!(s.prefill_load()[pw], 0);
        assert_eq!(s.decode_load()[h.decode_worker], 1);
        assert_eq!(h.prompt_tokens, 16);
        assert_eq!(h.n_blocks, 2, "16 tokens / 8 per block = 2 blocks");
        s.complete_decode(1).expect("decode done");
        assert_eq!(s.decode_load()[h.decode_worker], 0);
        assert_eq!(s.in_flight(), 0);
    }

    #[test]
    fn handoff_block_count_rounds_up() {
        let mut s = DisaggPdScheduler::new(1, 1, 4).expect("sched");
        s.schedule_prefill(&req(1, 9)).expect("prefill");
        let h = s.complete_prefill(1, 9).expect("handoff");
        assert_eq!(h.n_blocks, 3, "9 tokens / 4 = ceil 3 blocks");
    }

    #[test]
    fn stats_accumulate() {
        let mut s = DisaggPdScheduler::new(2, 2, 8).expect("sched");
        for i in 0..4 {
            s.schedule_prefill(&req(i, 8)).expect("prefill");
        }
        for i in 0..4 {
            s.complete_prefill(i, 8).expect("handoff");
        }
        for i in 0..4 {
            s.complete_decode(i).expect("decode");
        }
        let st = s.stats();
        assert_eq!(st.prefilled, 4);
        assert_eq!(st.handoffs, 4);
        assert_eq!(st.completed, 4);
        assert_eq!(st.blocks_migrated, 4, "each 8-token prompt = 1 block");
    }

    #[test]
    fn locate_tracks_phase_transitions() {
        let mut s = DisaggPdScheduler::new(1, 1, 8).expect("sched");
        s.schedule_prefill(&req(5, 8)).expect("prefill");
        assert_eq!(s.locate(5), Some((PdPhase::Prefill, 0)));
        s.complete_prefill(5, 8).expect("handoff");
        assert_eq!(s.locate(5), Some((PdPhase::Decode, 0)));
        s.complete_decode(5).expect("done");
        assert_eq!(s.locate(5), None);
    }

    #[test]
    fn decode_balances_across_workers() {
        let mut s = DisaggPdScheduler::new(1, 3, 8).expect("sched");
        // Three prompts → three handoffs spread over 3 decode workers.
        for i in 0..3 {
            s.schedule_prefill(&req(i, 8)).expect("prefill");
            s.complete_prefill(i, 8).expect("handoff");
        }
        assert_eq!(s.decode_load(), &[1, 1, 1], "decode load balanced");
    }

    #[test]
    fn empty_prompt_errors() {
        let mut s = DisaggPdScheduler::new(1, 1, 8).expect("sched");
        assert!(matches!(
            s.schedule_prefill(&req(1, 0)),
            Err(DistInferError::EmptyTokenSequence)
        ));
    }

    #[test]
    fn double_schedule_errors() {
        let mut s = DisaggPdScheduler::new(1, 1, 8).expect("sched");
        s.schedule_prefill(&req(1, 8)).expect("prefill");
        assert!(matches!(
            s.schedule_prefill(&req(1, 8)),
            Err(DistInferError::Internal(_))
        ));
    }

    #[test]
    fn complete_prefill_wrong_phase_errors() {
        let mut s = DisaggPdScheduler::new(1, 1, 8).expect("sched");
        s.schedule_prefill(&req(1, 8)).expect("prefill");
        s.complete_prefill(1, 8).expect("handoff"); // now in decode
        assert!(matches!(
            s.complete_prefill(1, 8),
            Err(DistInferError::Internal(_))
        ));
    }

    #[test]
    fn complete_decode_unknown_errors() {
        let mut s = DisaggPdScheduler::new(1, 1, 8).expect("sched");
        assert!(matches!(
            s.complete_decode(99),
            Err(DistInferError::Internal(_))
        ));
    }

    #[test]
    fn empty_pool_errors() {
        assert!(matches!(
            DisaggPdScheduler::new(0, 2, 8),
            Err(DistInferError::TooFewRanks { .. })
        ));
        assert!(matches!(
            DisaggPdScheduler::new(2, 0, 8),
            Err(DistInferError::TooFewRanks { .. })
        ));
    }
}

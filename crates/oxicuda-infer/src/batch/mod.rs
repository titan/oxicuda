//! Batching and scheduling subsystem.
//!
//! * [`sequence`]           — Sequence state machine, SamplingParams.
//! * [`scheduler`]          — FCFS scheduler with preemption.
//! * [`continuous_batcher`] — vLLM-style continuous batching orchestrator.
//! * [`chunked_prefill`]    — Sarathi chunked-prefill / decode piggyback planner.
//! * [`sampling_override`]  — per-sequence sampling-parameter overrides.

pub mod chunked_prefill;
pub mod continuous_batcher;
pub mod sampling_override;
pub mod scheduler;
pub mod sequence;

pub use chunked_prefill::{ChunkPlanner, ChunkedPrefillPlan, PrefillChunk, StepPacking};
pub use continuous_batcher::{BatcherConfig, ContinuousBatcher, GenerationOutput};
pub use sampling_override::{SamplingOverride, SamplingOverrideTable};
pub use scheduler::{ScheduledBatch, Scheduler, SchedulerConfig, StepResult};
pub use sequence::{FinishReason, SamplingParams, Sequence, SequenceId, SequenceStatus};

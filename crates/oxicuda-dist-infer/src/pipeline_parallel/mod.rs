//! Pipeline parallelism (PP) — the fourth parallelism axis.
//!
//! Pipeline parallelism partitions a model's *layers* into contiguous
//! **stages**, one per pipeline rank. A training/inference *micro-batch* flows
//! through the stages like an assembly line: stage `s` runs the forward pass on
//! micro-batch `m`, then hands its activations to stage `s + 1`. To keep all
//! stages busy, several micro-batches are kept in flight; the *order* in which
//! each stage runs forward (`F`) and backward (`B`) passes is the **pipeline
//! schedule**.
//!
//! This module is pure scheduling logic with exact oracles — no GPU required:
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`partition`] | [`partition::LayerPartition`] — balanced layer→stage assignment, incl. memory-aware variant |
//! | [`schedule`] | GPipe, 1F1B and interleaved-1F1B schedule generators + bubble accounting + hazard checks |
//!
//! # Schedules
//!
//! * **GPipe** (Huang 2019) — all forwards, then all backwards. Simple but the
//!   *pipeline bubble* (idle time) is `(p − 1)/m` of the runtime.
//! * **1F1B** (PipeDream / Megatron) — steady state alternates one forward and
//!   one backward, bounding activation memory to `p` micro-batches while keeping
//!   the same bubble ratio as GPipe but far less peak memory.
//! * **Interleaved 1F1B** (Narayanan 2021) — each rank owns `v` non-contiguous
//!   *model chunks*; the bubble shrinks to `(p − 1)/(m·v)`.
//!
//! # References
//! - Huang et al. (2019) "GPipe: Efficient Training of Giant Neural Networks
//!   using Pipeline Parallelism." NeurIPS.
//! - Narayanan et al. (2021) "Efficient Large-Scale Language Model Training on
//!   GPU Clusters Using Megatron-LM." SC.

pub mod partition;
pub mod schedule;

pub use partition::{LayerPartition, StageRange};
pub use schedule::{
    MicroBatchOp, OpKind, PipelineSchedule, gpipe_schedule, interleaved_1f1b_schedule,
    one_f_one_b_schedule,
};

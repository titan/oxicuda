//! Memory-efficiency utilities for PEFT: gradient checkpointing, activation recomputation.

/// Gradient checkpointing schedule and simulation.
pub mod grad_checkpoint;

pub use grad_checkpoint::{CheckpointConfig, CheckpointSchedule};

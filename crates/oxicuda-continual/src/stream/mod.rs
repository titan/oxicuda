//! Data stream abstractions for continual learning.
//!
//! Provides task-incremental and class-incremental stream interfaces
//! for sequential task delivery and evaluation.

pub mod class_stream;
pub mod cross_task_sampler;
pub mod scenario;
pub mod task_stream;

// ─── Scenario harness re-exports ──────────────────────────────────────────────
pub use scenario::{
    PermutedScenario, RotatedScenario, ScenarioConfig, SplitScenario, permuted_mnist,
    rotated_mnist, split_classes,
};

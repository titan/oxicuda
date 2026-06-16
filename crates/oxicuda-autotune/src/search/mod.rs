//! Schedule-space and loop-nest search strategies.
//!
//! This module hosts search algorithms that operate over *loop-nest schedules*
//! rather than the flat [`Config`](crate::config::Config) search space used by
//! the measurement-based tuners:
//!
//! - [`halide_schedule`] — a Halide-style (Adams et al., 2019) schedule search
//!   with a loop-nest IR, legality-checked transforms (tile/reorder/vectorize/
//!   parallelize/unroll), a feature-based cost model, and greedy/beam search.
//!
//! These searchers are pure-Rust and dependency-free.

pub mod halide_schedule;

pub use halide_schedule::{
    CostModel, FeatureCostModel, Loop, LoopNest, Schedule, ScheduleSearcher, Transform,
};

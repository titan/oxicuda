//! Analytical and learned cost models for autotuning.
//!
//! This module groups *predictive* models that estimate kernel performance
//! without running on hardware, complementing the measurement-based
//! [`BenchmarkEngine`](crate::benchmark::BenchmarkEngine):
//!
//! - [`roofline`] — the Williams (2009) roofline analytical bound, classifying
//!   a kernel as memory- or compute-bound from its arithmetic intensity and the
//!   device's peak compute and memory-bandwidth ceilings.
//! - [`latency_predictor`] — a learned ridge-regression surrogate that maps a
//!   feature vector extracted from a kernel/loop-nest descriptor to a predicted
//!   latency.
//!
//! Both models are pure-Rust and dependency-free; they are useful for pruning a
//! search space, seeding a search strategy, or scoring candidate schedules.

pub mod latency_predictor;
pub mod roofline;

pub use latency_predictor::{KernelDescriptor, LatencyPredictor, NUM_FEATURES};
pub use roofline::{BandwidthCeiling, Bound, Roofline, RooflineClassification};

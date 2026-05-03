//! One-shot supernet building blocks.
//!
//! - [`weight_share`] — `Supernet`: shared weights across all paths.
//! - [`path_sample`] — `PathSampler`: uniform and fairness-aware sampling.
//! - [`slimmable`] — `SlimmableNet`: width multipliers with adaptive BN stats.

pub mod path_sample;
pub mod slimmable;
pub mod weight_share;

pub use path_sample::{PathSampler, SamplingStrategy};
pub use slimmable::{BnStats, SlimmableNet, WIDTH_MULTIPLIERS};
pub use weight_share::Supernet;

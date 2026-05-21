//! One-shot supernet building blocks.
//!
//! - [`weight_share`] — `Supernet`: shared weights across all paths.
//! - [`path_sample`] — `PathSampler`: uniform and fairness-aware sampling.
//! - [`slimmable`] — `SlimmableNet`: width multipliers with adaptive BN stats.
//! - [`bignas`] — `BigNasSampler`: uniform sub-net + sandwich rule (Yu 2020 ECCV).

pub mod bignas;
pub mod path_sample;
pub mod slimmable;
pub mod weight_share;

pub use bignas::{BigNasConfig, BigNasSampler};
pub use path_sample::{PathSampler, SamplingStrategy};
pub use slimmable::{BnStats, SlimmableNet, WIDTH_MULTIPLIERS};
pub use weight_share::Supernet;

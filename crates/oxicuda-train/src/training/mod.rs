//! Higher-level training-loop infrastructure.
//!
//! This module collects host-side training utilities that orchestrate or
//! regularise the optimisation loop rather than performing the parameter
//! updates themselves:
//!
//! * [`crate::training::swa`] — Stochastic Weight Averaging and the companion SWALR schedule.
//! * [`crate::training::label_smoothing`] — label-smoothing cross-entropy loss and gradient.
//! * [`crate::training::early_stopping`] — metric-monitoring early-stop criterion.
//! * [`crate::training::curriculum`] — competence-based curriculum pacing over a sorted dataset.
//! * [`crate::training::sampler`] — data-loader index samplers (sequential, random, weighted,
//!   subset, batch).

/// Stochastic Weight Averaging (SWA) and SWALR (Izmailov et al., 2018).
pub mod swa;

/// Label-smoothing cross-entropy regularisation (Szegedy et al., 2016).
pub mod label_smoothing;

/// Early stopping on a monitored validation metric.
pub mod early_stopping;

/// Competence-based curriculum learning (Platanios et al., 2019).
pub mod curriculum;

/// Data-loader index samplers.
pub mod sampler;

pub use curriculum::{Curriculum, Pacing};
pub use early_stopping::{EarlyStopMode, EarlyStopping, EarlyStoppingConfig};
pub use label_smoothing::{LabelSmoothingConfig, LabelSmoothingCrossEntropy};
pub use sampler::{
    BatchSampler, RandomSampler, SequentialSampler, SubsetRandomSampler, WeightedRandomSampler,
};
pub use swa::{Swa, SwaLr, SwaLrMode};

//! Fairness-aware ranking and exposure-allocation utilities.

pub mod fairness_ranking;

pub use fairness_ranking::{FairnessRanker, FairnessRankerConfig, position_weight};

//! Logit-level distillation methods.

pub mod adaptive_temp;
pub mod decoupled_kd;
pub mod dist_distill;
pub mod hinton_kd;
pub mod skd;

pub use adaptive_temp::{AdaptiveTempConfig, AdaptiveTempScheduler, AdaptiveTempState};
pub use skd::{Skd, SkdConfig, SkdGatingMode};

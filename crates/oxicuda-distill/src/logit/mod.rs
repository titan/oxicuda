//! Logit-level distillation methods.

pub mod adaptive_temp;
pub mod decoupled_kd;
pub mod dist_distill;
pub mod hinton_kd;
pub mod logit_std;
pub mod skd;

pub use adaptive_temp::{AdaptiveTempConfig, AdaptiveTempScheduler, AdaptiveTempState};
pub use logit_std::{
    LogitStdConfig, logit_std_kd_loss, logit_std_kd_loss_batch, standardize, standardized_softmax,
};
pub use skd::{Skd, SkdConfig, SkdGatingMode};

//! Data-free distillation methods.

pub mod dafl;
pub mod dafl_deep;
pub mod dfad;
pub mod zskd;

pub use dafl_deep::{
    DeepGenerator, class_balance_entropy, conditional_one_hot_loss, generate_balanced_batch,
    label_balanced_classes,
};
pub use dfad::{Dfad, DfadConfig, DfadDims};

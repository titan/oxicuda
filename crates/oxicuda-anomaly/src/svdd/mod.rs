//! Deep SVDD (Ruff et al. 2018) — One-Class Deep Learning.
pub mod deep_svdd;
pub mod soft_svdd;
pub mod trainable_svdd;

pub use trainable_svdd::{
    TrainableSvddConfig, TrainableSvddFit, trainable_svdd_fit, trainable_svdd_loss_history,
    trainable_svdd_predict, trainable_svdd_score,
};

//! Feature-level distillation methods.

pub mod at;
pub mod crd_multi;
pub mod fitnets;
pub mod mgd;
pub mod pkt;
pub mod projection_free;
pub mod self_kd;
pub mod tinybert;

pub use crd_multi::{CrdMemoryBank, CrdMultiConfig, CrdMultiLoss};
pub use mgd::{MgdConfig, MgdGenerator, forward_generator, generate_mask, mgd_loss};
pub use projection_free::{ProjFreeConfig, ProjFreeDistiller, ProjFreeLossType, ProjFreeNorm};
pub use self_kd::{MixupBatchElement, SelfKd, SelfKdConfig};
pub use tinybert::{
    TinyBertConfig, TinyBertGeneralLoss, TinyBertProjection, attention_mse, embedding_mse,
    hidden_mse, mlp2_forward, prediction_loss,
};

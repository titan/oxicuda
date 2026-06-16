//! Feature-level distillation methods.

pub mod at;
pub mod crd_multi;
pub mod cwd;
pub mod fitnets;
pub mod mgd;
pub mod ofd;
pub mod pkt;
pub mod projection_free;
pub mod review_kd;
pub mod self_kd;
pub mod simkd;
pub mod tinybert;

pub use crd_multi::{CrdMemoryBank, CrdMultiConfig, CrdMultiLoss};
pub use cwd::{
    ChannelProjector, CwdConfig, channel_kl, cwd_loss, cwd_loss_projected, spatial_softmax,
};
pub use mgd::{MgdConfig, MgdGenerator, forward_generator, generate_mask, mgd_loss};
pub use ofd::{OfdConnector, estimate_margins, margin_relu, ofd_loss, ofd_loss_batch, partial_l2};
pub use projection_free::{ProjFreeConfig, ProjFreeDistiller, ProjFreeLossType, ProjFreeNorm};
pub use review_kd::{AbfModule, FeatureMap, avg_pool, hcl_loss, review_connection};
pub use self_kd::{MixupBatchElement, SelfKd, SelfKdConfig};
pub use simkd::{SimKdProjector, TeacherClassifier, simkd_forward, simkd_loss};
pub use tinybert::{
    TinyBertConfig, TinyBertGeneralLoss, TinyBertProjection, attention_mse, embedding_mse,
    hidden_mse, mlp2_forward, prediction_loss,
};

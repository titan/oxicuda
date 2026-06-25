//! Distillation loss functions.

pub mod attention_transfer;
pub mod cross_modal;
pub mod distwrd;
pub mod fitnets;
pub mod kd_loss;
pub mod minkd;
pub mod nst;
pub mod qat_distill;
pub mod sp;
pub mod tinybert_loss;
pub mod vid;
pub use attention_transfer::{attention_map, attention_transfer_loss};
pub use cross_modal::{
    AlignDistance, CrossModalConfig, CrossModalProjector, cross_modal_contrastive_loss,
    cross_modal_loss, paired_alignment_loss,
};
pub use distwrd::{DistWrd, wasserstein1_cdf};
pub use fitnets::fitnet_hint_loss;
pub use kd_loss::{KdLoss, KdLossConfig};
pub use minkd::MinKd;
pub use nst::{NstKernel, normalize_channels, nst_loss};
pub use qat_distill::{
    AffineQuantParams, Fp8Format, QatDistillConfig, QuantKind, fake_quant_affine, fake_quant_fp8,
    qat_kd_loss, qat_kd_loss_batch, quantization_error, quantize_student, ste_grad_mask,
};
pub use sp::{similarity_matrix, sp_loss};
pub use tinybert_loss::{TinyBertLoss, TinyBertLossConfig};
pub use vid::{VidRegressor, softplus};

//! Distillation loss functions.

pub mod attention_transfer;
pub mod distwrd;
pub mod fitnets;
pub mod kd_loss;
pub mod minkd;
pub mod nst;
pub mod sp;
pub mod tinybert_loss;
pub mod vid;
pub use attention_transfer::{attention_map, attention_transfer_loss};
pub use distwrd::{DistWrd, wasserstein1_cdf};
pub use fitnets::fitnet_hint_loss;
pub use kd_loss::{KdLoss, KdLossConfig};
pub use minkd::MinKd;
pub use nst::{NstKernel, normalize_channels, nst_loss};
pub use sp::{similarity_matrix, sp_loss};
pub use tinybert_loss::{TinyBertLoss, TinyBertLossConfig};
pub use vid::{VidRegressor, softplus};

//! Cross-attention and self-cross-attention block implementations.

pub mod cross_attention;
pub mod flamingo;
pub(crate) mod masked_mha;
pub mod self_cross_block;

pub use flamingo::{FlamingoGatedConfig, FlamingoGatedLayer, FlamingoGatedWeights};

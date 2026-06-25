//! Vision Transformer (ViT) components.
//!
//! Provides:
//! - **`ViTBlock`**: pre-norm transformer block (MHSA + MLP with GELU).
//! - **`ViTEncoder`**: stack of `depth` ViT blocks with a final layer-norm.
//! - **`ViTModel`**: full ViT pipeline (patch embed → CLS prepend →
//!   positional encoding → encoder → classification head).
//! - **`drop_path`**: DropPath / stochastic-depth residual regularisation with
//!   a linear depth schedule.
//! - **`flash_attention`**: FlashAttention-2 online-softmax CPU reference.
//! - **`cait`**: CaiT Class-Attention layers + LayerScale.
//! - **`xcit`**: XCiT Cross-Covariance Attention (channel-wise, linear in tokens).
//! - **`t2t`**: Tokens-to-Token soft-split re-tokenization.
//! - **`eva`**: EVA / EVA-CLIP variant configs + mean-pool projection head.
//! - **`quantize`**: INT8 post-training quantisation (per-channel weights,
//!   dynamic activations, integer-domain linear).

pub mod cait;
pub mod drop_path;
pub mod eva;
pub mod flash_attention;
pub mod mae;
pub mod quantize;
pub mod swin;
pub mod t2t;
pub mod vit_block;
pub mod vit_encoder;
pub mod vit_model;
pub mod vit_patch;
pub mod xcit;

pub use cait::{CaitConfig, ClassAttention, ClassAttentionStack, LayerScale};
pub use drop_path::{DropPath, DropPathConfig, drop_path_schedule};
pub use eva::{EvaPoolHead, EvaVariant};
pub use flash_attention::{FlashAttnConfig, flash_attention, reference_attention};
pub use mae::{Mae, MaeConfig, MaskMeta, generate_random_mask, mae_loss};
pub use quantize::{
    QuantLinear, QuantParams, QuantWeight, dequantize_tensor, fake_quantize_symmetric,
    quantize_tensor_affine, quantize_tensor_symmetric,
};
pub use swin::{SwinBlock, SwinConfig, SwinWeights};
pub use t2t::{SoftSplitConfig, T2tModule, soft_split, tokens_to_map};
pub use vit_block::{ViTBlock, ViTBlockConfig, ViTBlockWeights};
pub use vit_encoder::{ViTEncoder, ViTEncoderConfig};
pub use vit_model::{ViTConfig, ViTModel, ViTModelWeights};
pub use vit_patch::{VitPatchConfig, VitPatchEmbed};
pub use xcit::{XcitBlock, XcitConfig, cross_covariance_attention};

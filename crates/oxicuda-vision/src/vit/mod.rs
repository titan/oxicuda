//! Vision Transformer (ViT) components.
//!
//! Provides:
//! - **`ViTBlock`**: pre-norm transformer block (MHSA + MLP with GELU).
//! - **`ViTEncoder`**: stack of `depth` ViT blocks with a final layer-norm.
//! - **`ViTModel`**: full ViT pipeline (patch embed → CLS prepend →
//!   positional encoding → encoder → classification head).

pub mod mae;
pub mod swin;
pub mod vit_block;
pub mod vit_encoder;
pub mod vit_model;
pub mod vit_patch;

pub use mae::{Mae, MaeConfig, MaskMeta, generate_random_mask, mae_loss};
pub use swin::{SwinBlock, SwinConfig, SwinWeights};
pub use vit_block::{ViTBlock, ViTBlockConfig, ViTBlockWeights};
pub use vit_encoder::{ViTEncoder, ViTEncoderConfig};
pub use vit_model::{ViTConfig, ViTModel, ViTModelWeights};
pub use vit_patch::{VitPatchConfig, VitPatchEmbed};

//! Patch embedding for Vision Transformers.
//!
//! Converts a CHW image to a sequence of patch tokens by applying a
//! strided Conv2D with `kernel_size == stride == patch_size`.
//! Also provides 2-D sinusoidal and learnable positional encodings.

pub mod conv2d_patch;
pub mod pos_embed;

pub use conv2d_patch::{PatchEmbed, PatchEmbedConfig, PatchEmbedWeights, prepend_cls};
pub use pos_embed::{LearnablePosEmbed, add_pos_embed, pos_2d_sincos};

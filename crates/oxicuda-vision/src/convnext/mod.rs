//! ConvNeXt modern-CNN components.
//!
//! Provides:
//! - **`ConvNextBlock`**: depthwise 7×7 same-pad convolution + channel
//!   LayerNorm + 1×1 inverted-bottleneck expansion → GELU → 1×1 projection +
//!   per-channel layer scale + residual (Liu et al. 2022 CVPR).

pub mod block;

pub use block::{ConvNextBlock, ConvNextConfig};

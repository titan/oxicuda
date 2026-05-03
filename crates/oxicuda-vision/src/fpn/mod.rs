//! Feature Pyramid Network (FPN) components.
//!
//! Provides:
//! - **`LateralConv1x1`**: 1×1 lateral convolution for channel reduction.
//! - **`Fpn`**: top-down feature pyramid with nearest-neighbour upsampling
//!   and 3×3 smoothing convolutions.

pub mod lateral;
pub mod top_down;

pub use lateral::{LateralConv1x1, LateralWeights};
pub use top_down::{FeatureMap, Fpn, FpnConfig};

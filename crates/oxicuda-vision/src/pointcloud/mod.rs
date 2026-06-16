//! Point-cloud neural network primitives.
//!
//! Provides:
//! - **`point_transformer`**: the Point Transformer vector self-attention layer
//!   (Zhao et al. 2021) — subtraction-relation attention over kNN
//!   neighbourhoods with a learned relative-position encoding.

pub mod point_transformer;

pub use point_transformer::{PointAttention, PointTransformerConfig, PointTransformerLayer};

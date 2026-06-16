//! Image segmentation models.
//!
//! Provides:
//! - **`sam`**: a compact, faithful CPU reference of the *Segment Anything
//!   Model* (Kirillov et al. 2023) — ViT image encoder, a point/box/mask prompt
//!   encoder with random-Fourier positional embeddings, and a two-way
//!   transformer mask decoder predicting masks plus IoU quality scores.

pub mod sam;

pub use sam::{
    ImageEncoder, MaskDecoder, MaskPrediction, MultiHeadAttention, PositionEmbeddingRandom,
    PromptEncoder, Sam, SamConfig, TwoWayAttentionBlock, TwoWayBlockOutput, TwoWayTransformer,
};

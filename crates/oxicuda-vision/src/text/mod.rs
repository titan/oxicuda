//! Text encoders for vision-language models.
//!
//! Provides:
//! - **`clip_text`**: the CLIP Transformer text tower (Radford et al. 2021) —
//!   token + positional embeddings, pre-LN causal self-attention blocks, a
//!   final LayerNorm, EOS-token pooling, and a linear projection into the
//!   joint image-text embedding space with L2 normalisation.

pub mod clip_text;

pub use clip_text::{ClipTextConfig, ClipTextEncoder};

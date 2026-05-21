//! Transformer layer building blocks.
//!
//! # Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`attention`] | Multi-head attention (MHA/GQA) with KV-cache |
//! | [`embedding`] | Token embedding, learned positional embedding, RoPE |
//! | [`ffn`]       | MLP (GELU) and SwiGLU feed-forward networks |
//! | [`mod@flash_attention`] | Tiled online-softmax FlashAttention CPU reference |
//! | [`norm`]      | RMSNorm and LayerNorm |
//! | [`rope_scaling`] | Long-context RoPE scaling (Linear / NTK-aware / YaRN) |
//! | [`transformer`] | GPT-2 and LLaMA transformer blocks; `PastKvCache` |

pub mod attention;
pub mod embedding;
pub mod ffn;
pub mod flash_attention;
pub mod norm;
pub mod rope_scaling;
pub mod transformer;

pub use attention::{LayerKvCache, MultiHeadAttention};
pub use embedding::{LearnedPositionalEmbedding, RotaryEmbedding, TokenEmbedding};
pub use ffn::{MlpFfn, SwiGluFfn, gelu, silu};
pub use flash_attention::{FlashAttentionConfig, flash_attention};
pub use norm::{LayerNorm, RmsNorm};
pub use rope_scaling::{RopeScaling, RopeScalingKind};
pub use transformer::{GptBlock, LlamaBlock, PastKvCache};

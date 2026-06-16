//! Hybrid Mamba–Attention architectures.
//!
//! Interleaves SSM (Mamba) and self-attention layers according to a
//! configurable schedule, following the Jamba-style hybrid design.

pub mod mamba_attn;
pub use mamba_attn::{HybridBlock, HybridConfig};

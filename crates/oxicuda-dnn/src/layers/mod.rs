//! CPU-side layer implementations.
//!
//! This module provides pure-Rust CPU simulations of several deep-learning
//! layers that complement the GPU-accelerated primitives elsewhere in the
//! crate.  They are suitable for prototyping, unit-testing, and environments
//! without CUDA hardware.
//!
//! | Submodule | Description |
//! |-----------|-------------|
//! | [`moe`] | Mixture-of-Experts (Shazeer 2017) CPU layer |
//! | [`flash_attn`] | FlashAttention-2 tiling CPU simulation (Dao 2022) |
//! | [`rwkv`] | RWKV linear-attention recurrent layer (Peng 2023) |

pub mod flash_attn;
pub mod moe;
pub mod rwkv;

pub use flash_attn::{FlashAttention, FlashAttnConfig};
pub use moe::{MoeConfig, MoeLayer};
pub use rwkv::{RwkvConfig, RwkvLayer};

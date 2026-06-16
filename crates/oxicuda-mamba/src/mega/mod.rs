//! MEGA — Moving Average Equipped Gated Attention (Ma et al. 2022).
//!
//! This module provides [`MegaBlock`]: a sequence layer that combines an
//! Exponential Moving Average (EMA) sub-layer for local context with a
//! single-headed gated attention mechanism for long-range dependencies.

pub mod mega_block;

pub use mega_block::{MegaBlock, MegaConfig};

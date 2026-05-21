//! RWKV (Receptance Weighted Key Value) architecture modules.
//!
//! Implements the RWKV-4 architecture (Peng et al., 2023) in pure Rust.
//!
//! # Submodules
//!
//! - [`time_mixing`]   — WKV (Weighted Key-Value) attention-free time-mixing operation.
//! - [`channel_mixing`] — Gated FFN with Square-ReLU channel-mixing operation.
//! - [`rwkv_block`]    — Complete RWKV residual block combining both operations.

pub mod channel_mixing;
pub mod rwkv5;
pub mod rwkv_block;
pub mod time_mixing;

//! Mamba (S6) architecture: selective scan, block, and full language model.
//!
//! # Submodules
//!
//! - [`selective_scan`] — Pure-Rust S6 selective scan reference implementation.
//! - [`mamba_block`]    — Single Mamba residual block with all helper ops.
//! - [`mamba_model`]    — Full Mamba language model (embedding + layers + LM head).

pub mod mamba_block;
pub mod mamba_model;
pub mod selective_scan;

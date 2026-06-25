//! Mamba (S6) architecture: selective scan, block, and full language model.
//!
//! # Submodules
//!
//! - [`selective_scan`]          — Pure-Rust S6 selective scan reference (sequential).
//! - [`selective_scan_parallel`] — Work-efficient Blelloch parallel-scan model of
//!   the same recurrence (the algorithm the fused GPU kernel realises).
//! - [`selective_scan_mixed`]    — FP16 / BF16 mixed-precision scan with an FP32
//!   accumulator (tensor-core numerics on the CPU).
//! - [`mamba_block`]             — Single Mamba residual block with all helper ops.
//! - [`mamba_model`]             — Full Mamba language model (embedding + layers + LM head).

pub mod mamba_block;
pub mod mamba_model;
pub mod selective_scan;
pub mod selective_scan_mixed;
pub mod selective_scan_parallel;

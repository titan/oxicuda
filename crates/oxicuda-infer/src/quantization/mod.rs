//! Weight-quantization algorithms for low-bit LLM inference.
//!
//! # Modules
//!
//! | Module | Algorithm |
//! |--------|-----------|
//! | [`awq`] | Activation-aware Weight Quantization (Lin et al. 2023) |
//!
//! These are pure-Rust CPU reference implementations of post-training
//! weight-only quantizers. Each operates on a row-major weight matrix, produces
//! integer codes with per-group affine parameters, and offers an exact
//! dequantization path for round-trip validation.

pub mod awq;

pub use awq::{
    AwqConfig, AwqResult, GroupParams, awq_dequantize, awq_output_mse, awq_quantize,
    dense_output_mse, group_dequantize, group_quantize,
};

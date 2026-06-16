//! Quantisation utilities for distributed GPU inference.
//!
//! ## Modules
//!
//! * `fp8_infer` — FP8 (E4M3 / E5M2) per-tensor and block-wise
//!   quantisation/dequantisation.

pub mod fp8_infer;

pub use fp8_infer::{
    Fp8Format, dequantize_fp8, dequantize_fp8_block, quantize_fp8, quantize_fp8_block,
};

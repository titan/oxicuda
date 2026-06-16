//! Quantization methods for weight compression in PEFT: NF4 and FP4 (e2m1),
//! plus block-wise 8-bit Adam optimizer state.

/// Block-wise 8-bit Adam optimizer with INT8-quantized moment buffers (Dettmers et al. 2022).
pub mod adam8bit;
/// NF4 and FP4 quantization implementations.
pub mod nf4_quant;

pub use adam8bit::{Adam8bit, Adam8bitConfig, BlockwiseInt8};
pub use nf4_quant::{NF4_QUANTS, dequantize_fp4, dequantize_nf4, quantize_fp4, quantize_nf4};

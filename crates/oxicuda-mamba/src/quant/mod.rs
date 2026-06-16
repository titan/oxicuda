//! Post-training quantization for Mamba / SSM weights.
//!
//! # Submodules
//!
//! - [`qmamba`] — Q-Mamba symmetric INT8 post-training quantization of linear
//!   projection matrices and SSM parameters, with a dequantized forward.

pub mod qmamba;

pub use qmamba::{QMambaQuantizer, QuantScheme, QuantizedTensor};

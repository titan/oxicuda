//! INT8 post-training quantisation inference path for time-series models.
//!
//! Symmetric (zero-point-free) linear quantisation of weight matrices and
//! activations for TCN / PatchTST inference acceleration, with per-tensor and
//! per-output-channel granularity. See [`int8`] for the [`QuantLinear`] layer
//! and quantisation-error utilities.

pub mod int8;

pub use int8::{
    QuantConfig, QuantGranularity, QuantLinear, dequantise_tensor, quantise_tensor,
    relative_quant_error,
};

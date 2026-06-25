//! CPU noise-generation kernels for DP-SGD-style privatisation.
//!
//! These are the CPU reference implementations of the noise/clip kernels that a
//! GPU backend fuses on-device:
//!
//! - [`mixed_precision`] — sample Gaussian noise, store it in IEEE-754 binary16
//!   (FP16) and accumulate in FP32 ("FP16 sample, FP32 accumulate"), including a
//!   complete pure-Rust FP16 round-trip with round-to-nearest-even, subnormals,
//!   overflow and underflow.
//! - [`fused_clip_noise`] — fuse per-vector L2 gradient clipping and Gaussian
//!   noise addition into a single pass over the gradient (saving one
//!   global-memory pass), verified bit-for-bit against the two-pass reference.

pub mod fused_clip_noise;
pub mod mixed_precision;

pub use fused_clip_noise::{
    fused_clip_and_noise, fused_clip_and_noise_in_place, sequential_clip_then_noise,
};
pub use mixed_precision::{
    MixedPrecisionNoise, add_fp16_noise_fp32_accumulate, f16_bits_to_f32, f32_to_f16_bits,
    mixed_precision_gaussian, quantize_f16,
};

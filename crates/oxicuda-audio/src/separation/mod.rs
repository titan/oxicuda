//! Audio source separation models for `oxicuda-audio`.
//!
//! Provides:
//! - **`conv_tasnet`**: Conv-TasNet (Luo & Mesgarani 2019) fully convolutional
//!   time-domain audio source separation with learned encoder/decoder and
//!   multi-block temporal convolutional network (TCN) separator.

pub mod conv_tasnet;

pub use conv_tasnet::{
    ConvTasNet, ConvTasNetConfig, ConvTasNetWeights, SeparationResult, TcnBlockWeights,
};

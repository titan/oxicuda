//! Audio source separation models for `oxicuda-audio`.
//!
//! Provides:
//! - **`conv_tasnet`**: Conv-TasNet (Luo & Mesgarani 2019) fully convolutional
//!   time-domain audio source separation with learned encoder/decoder and
//!   multi-block temporal convolutional network (TCN) separator.
//! - **`hpss`**: Harmonic/Percussive Source Separation via median filtering
//!   (Fitzgerald 2010 DAFX).

pub mod conv_tasnet;
pub mod hpss;

pub use conv_tasnet::{
    ConvTasNet, ConvTasNetConfig, ConvTasNetWeights, SeparationResult, TcnBlockWeights,
};
pub use hpss::{HpssConfig, HpssMask, HpssResult, hpss, hpss_masks, median_filter_1d};

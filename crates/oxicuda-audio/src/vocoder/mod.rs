//! WaveNet-style neural vocoder for `oxicuda-audio`.
//!
//! Provides a stack of dilated residual blocks that implement the WaveNet
//! architecture (van den Oord et al., 2016), suitable for neural waveform
//! synthesis from conditioning features.
//!
//! The primary sub-modules are:
//! - **`wavenet_block`**: A single dilated causal residual block with gated
//!   activation and separate skip / residual projections.
//! - **`dilated_stack`**: A full multi-cycle stack of `WaveNetBlock`s that
//!   accumulates skip outputs and applies a two-layer output head.
//! - **`griffin_lim`**: Griffin-Lim Algorithm and Fast Griffin-Lim (FGLA) for
//!   phase reconstruction from a magnitude spectrogram (Griffin & Lim 1984;
//!   Perraudin 2013).

pub mod dilated_stack;
pub mod griffin_lim;
pub mod hifigan;
pub mod wavenet_block;

pub use dilated_stack::{WaveNetConfig, WaveNetStack};
pub use griffin_lim::{
    GriffinLimConfig, griffin_lim, istft_hann, magnitude_from_signal, stft_hann,
};
pub use hifigan::{HifiGanConfig, HifiGanGenerator, HifiGanWeights, ResBlockWeights};
pub use wavenet_block::WaveNetBlock;

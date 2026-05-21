//! WaveNet-style neural vocoder for `oxicuda-audio`.
//!
//! Provides a stack of dilated residual blocks that implement the WaveNet
//! architecture (van den Oord et al., 2016), suitable for neural waveform
//! synthesis from conditioning features.
//!
//! The two primary sub-modules are:
//! - **`wavenet_block`**: A single dilated causal residual block with gated
//!   activation and separate skip / residual projections.
//! - **`dilated_stack`**: A full multi-cycle stack of `WaveNetBlock`s that
//!   accumulates skip outputs and applies a two-layer output head.

pub mod dilated_stack;
pub mod hifigan;
pub mod wavenet_block;

pub use dilated_stack::{WaveNetConfig, WaveNetStack};
pub use hifigan::{HifiGanConfig, HifiGanGenerator, HifiGanWeights, ResBlockWeights};
pub use wavenet_block::WaveNetBlock;

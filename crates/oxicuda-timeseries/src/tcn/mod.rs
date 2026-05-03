//! Temporal Convolutional Network (TCN) encoder.
//!
//! Stacks multiple [`TcnBlock`]s with exponentially doubling dilations to
//! achieve a large receptive field with efficient parallelism.

pub mod tcn_encoder;
pub mod temporal_block;

pub use tcn_encoder::{TcnConfig, TcnEncoder};
pub use temporal_block::TcnBlock;

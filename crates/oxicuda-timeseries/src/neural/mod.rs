//! Neural forecasting building blocks for `oxicuda-timeseries`.
//!
//! This module provides a simpler, standalone N-BEATS implementation that
//! stores per-block, per-layer weights as flat `Vec<f32>` buffers and performs
//! pure-Rust forward passes with double-residual stacking.

pub mod nbeats;
pub use nbeats::{BasisType, Nbeats, NbeatsBlock, NbeatsConfig};

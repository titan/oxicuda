//! TimesNet: 2-D temporal variation modelling for time-series forecasting.
//!
//! Implements Wu et al. 2023 "TimesNet: Temporal 2D-Variation Modeling for
//! General Time Series Analysis" (ICLR 2023) as a pure-Rust CPU reference.
//!
//! ## Module layout
//!
//! - [`times_block`] — single `TimesBlock` with FFT period detection and
//!   depthwise 2-D convolution.
//! - [`timesnet`] — full `TimesNet` model with input projection, stacked
//!   blocks, and prediction head.

pub mod times_block;
#[allow(clippy::module_inception)]
pub mod timesnet;

pub use times_block::TimesBlock;
pub use timesnet::{TimesNet, TimesNetConfig};

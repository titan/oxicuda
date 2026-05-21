//! N-BEATS — Neural Basis Expansion Analysis for Time Series.
//!
//! Implements the hierarchical double-residual stacking architecture from
//! Oreshkin et al. (ICLR 2020).  Three block types are supported:
//!
//! - **Generic** — unconstrained learned linear expansion heads.
//! - **Trend** — polynomial basis functions for trend modelling.
//! - **Seasonality** — Fourier (cos + sin) basis for periodic patterns.
//!
//! Each block produces a backcast (subtracted from the running residual)
//! and a forecast (accumulated into the final output).

#[allow(clippy::module_inception)]
pub mod nbeats;
pub mod nbeats_block;

pub use nbeats::{NBeats, NBeatsConfig};
pub use nbeats_block::{NBeatsBlock, NBeatsBlockType};

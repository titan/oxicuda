//! Plot-friendly helpers for survival analysis outputs.
//!
//! Provides step-function array types and conversion utilities compatible
//! with matplotlib's `step` rendering format.

pub mod step_functions;
pub use step_functions::*;

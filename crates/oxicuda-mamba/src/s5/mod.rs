//! S5 (Simplified State Space Layer) architecture.
//!
//! S5 (Smith et al. 2022 "Simplified State Space Layers for Sequence Modeling")
//! simplifies S4 by replacing the DPLR-structured A matrix with a plain
//! **diagonal real A** matrix.  This allows closed-form ZOH discretization
//! for each diagonal element independently, yielding a fully MIMO
//! (Multi-Input Multi-Output) recurrent layer.
//!
//! ## Submodules
//!
//! - [`s5_layer`] — `S5Config`, `S5Weights`, `S5Layer`: full MIMO S5 sequence layer.

pub mod s5_layer;
pub use s5_layer::{S5Config, S5Layer, S5Weights};

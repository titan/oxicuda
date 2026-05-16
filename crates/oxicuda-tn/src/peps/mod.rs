//! Projected Entangled Pair States (PEPS) — 2D tensor networks.

pub mod contraction;
pub mod peps;

pub use contraction::boundary_mps_contraction;
pub use peps::{Peps, PepsTensor};

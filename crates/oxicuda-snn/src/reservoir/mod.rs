//! Reservoir-computing primitives: random recurrent reservoirs of spiking neurons.

/// Liquid State Machine — Maass et al. 2002.
pub mod lsm;

/// Echo State Network — Jaeger 2001, leaky integrator rate-coded reservoir.
pub mod esn;
pub use esn::{Esn, EsnConfig, EsnState, ridge_regression};

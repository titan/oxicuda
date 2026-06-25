//! Reservoir-computing primitives: random recurrent reservoirs of spiking neurons.

/// Adaptive spectral-radius scheduling / control during training.
pub mod adaptive_spectral;

/// Multi-reservoir hierarchical (deep) Liquid State Machine.
pub mod hierarchical_lsm;

/// Liquid Time-Constant network — Hasani et al. 2021 (analog, reservoir-related).
pub mod ltc;

/// Liquid State Machine — Maass et al. 2002.
pub mod lsm;

/// Echo State Network — Jaeger 2001, leaky integrator rate-coded reservoir.
pub mod esn;
pub use esn::{Esn, EsnConfig, EsnState, ridge_regression};

/// Online RLS / FORCE ridge-regression readout (Sussillo & Abbott 2009).
pub mod ridge_readout;
pub use ridge_readout::RidgeReadout;

//! Generative models over 3D point clouds.
//!
//! Currently provides the PointFlow-style continuous-normalizing-flow (CNF)
//! core ([`pointflow`]): an exactly-invertible flow with an exact
//! change-of-variables log-density, rigorously verifiable on CPU
//! (invertibility, log-det, density normalization). The velocity field is
//! untrained — no generated-shape realism is claimed.

pub mod pointflow;

pub use pointflow::{CnfConfig, ContinuousNormalizingFlow, PointFlowModel};

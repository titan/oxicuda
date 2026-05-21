//! Jordan-Kinderlehrer-Otto (JKO) proximal scheme for Wasserstein gradient flows.
//!
//! Time-discretises a gradient flow on the space of probability measures with
//! respect to the 2-Wasserstein metric: at each step the new density is the
//! argmin of `(τ/2)·OT(ρ_new, ρ) + F(ρ_new)`, where `F` is the driving free
//! energy functional. The classic example is the heat equation, recovered when
//! `F(ρ) = ε · Σ ρ log ρ` is the (negative) entropy.

/// JKO proximal step for Wasserstein gradient flows on the entropy and external
/// potential energies.
pub mod jko;
/// JKO particle gradient flow via the Blob method (Carrillo et al. 2019).
pub mod jko_step;

pub use jko_step::{
    JkoState, JkoStepConfig, jko_init, jko_run, jko_step, jko_wasserstein_distance,
};

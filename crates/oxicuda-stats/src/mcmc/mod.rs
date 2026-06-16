//! Gradient-based Markov-chain Monte Carlo samplers.
//!
//! This module provides geometry-aware MCMC algorithms that exploit the gradient
//! of the log-target to make long, high-acceptance moves through parameter space:
//!
//! - [`hmc`] — Hamiltonian Monte Carlo with a leapfrog integrator (Neal 2011).
//! - [`nuts`] — the No-U-Turn Sampler, which auto-tunes the trajectory length by
//!   building a binary tree of leapfrog steps until a U-turn (Hoffman & Gelman 2014).
//!
//! Both samplers operate on a [`hmc::PotentialTarget`], which wraps a
//! user-supplied potential energy `U(q) = −log π(q)` together with an optional
//! analytic gradient (a central finite-difference fallback is provided).
//!
//! # References
//! - Neal, R. M. (2011). "MCMC using Hamiltonian dynamics." *Handbook of Markov
//!   Chain Monte Carlo*, Ch. 5.
//! - Hoffman, M. D. & Gelman, A. (2014). "The No-U-Turn Sampler." *JMLR*
//!   15:1593-1623.

pub mod hmc;
pub mod nuts;

pub use hmc::{
    HmcConfig, HmcSamples, PotentialTarget, hamiltonian, hmc_sample, leapfrog, leapfrog_step,
};
pub use nuts::{NutsConfig, NutsSamples, no_u_turn, nuts_sample};

//! Spiking neuron models.
//!
//! Each module exposes a `*Config`, `*State`, and a step function that updates
//! the state in place and writes binary spikes to an output buffer. All numerics
//! are `f32`; integration uses explicit Euler unless otherwise noted.

/// Adaptive Exponential Integrate-and-Fire neuron (Brette-Gerstner 2005).
pub mod adex;
/// Pure Integrate-and-Fire neuron (no leak).
pub mod integrate_fire;
/// Izhikevich neuron (2003) with quadratic + recovery dynamics.
pub mod izhikevich;
/// Leaky Integrate-and-Fire neuron.
pub mod lif;
/// Stochastic Poisson rate neuron.
pub mod poisson;

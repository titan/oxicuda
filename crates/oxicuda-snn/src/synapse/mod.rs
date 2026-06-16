//! Synaptic models for spiking neural networks.
//!
//! Each submodule provides a `*Config`, `*State`, scalar `step`, and slice-level
//! batch routine following the same shape as the [`crate::neuron`] modules. All
//! synaptic kernels operate purely on `f64` host buffers; PTX-side counterparts
//! live next to the other GPU kernels in [`crate::ptx_kernels`].

/// Alpha-function synapse (Rall 1967): finite-rise post-synaptic current.
pub mod alpha;
/// Conductance-based exponential synapses: CUBA (current-based) and COBA
/// (conductance-based) variants in the sense of Dayan & Abbott (2001).
pub mod conductance;
/// Integer-step axonal/synaptic delay lines (Izhikevich polychronization 2006).
pub mod delay;
/// Tsodyks-Markram short-term plasticity: facilitation and depression.
pub mod tsodyks_markram;

pub use alpha::{AlphaConfig, AlphaState, alpha_decay, alpha_step, alpha_step_batch};
pub use delay::{DelayBank, DelayConfig, DelayLine};
pub use tsodyks_markram::{TmConfig, TmState, tm_step};

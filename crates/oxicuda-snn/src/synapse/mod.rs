//! Synaptic models for spiking neural networks.
//!
//! Each submodule provides a `*Config`, `*State`, scalar `step`, and slice-level
//! batch routine following the same shape as the [`crate::neuron`] modules. All
//! synaptic kernels operate purely on `f64` host buffers; PTX-side counterparts
//! live next to the other GPU kernels in [`crate::ptx_kernels`].

/// Conductance-based exponential synapses: CUBA (current-based) and COBA
/// (conductance-based) variants in the sense of Dayan & Abbott (2001).
pub mod conductance;

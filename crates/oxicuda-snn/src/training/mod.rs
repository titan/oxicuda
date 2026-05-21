//! Surrogate-gradient supervised training algorithms for spiking neural networks.
//!
//! This module groups three closely-related but distinct algorithms used to train
//! SNNs end-to-end with gradient descent despite the non-differentiable spike
//! function:
//!
//! * [`bptt`] — vanilla Backpropagation-Through-Time with a smooth surrogate
//!   for `dS/dV` and an analytical hard/soft reset gradient.
//! * [`stbp`] — Spatio-Temporal Backpropagation (Wu et al. 2018) with an
//!   explicit `(1 − S_t)` reset-gating factor on the recurrent membrane gradient.
//! * [`slayer`] — Spike-LAYer Error Reassignment (Shrestha & Orchard 2018) using
//!   a low-pass-filtered post-synaptic-potential kernel and an `MSE` on the
//!   filtered output spike train.

/// Backprop-through-time for SNN with surrogate gradients.
pub mod bptt;
/// e-prop online learning rule (Bellec 2020) and DECOLLE variant (Kaiser 2020).
pub mod eprop;
/// SLAYER spike layer error reassignment.
pub mod slayer;
/// Spatio-temporal backprop with explicit reset gradient (Wu et al. 2018).
pub mod stbp;

pub use eprop::{
    EligibilityTraces, EpropConfig, LearningSignal, apply_weight_update, compute_weight_update,
    decolle_learning_signals, eprop_step, update_eligibility_traces, update_running_rates,
};

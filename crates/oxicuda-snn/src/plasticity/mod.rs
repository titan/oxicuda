//! Spike-timing-dependent plasticity (STDP) rules for online learning.
//!
//! All rules in this module are *online* and *local*: weights are updated
//! incrementally each timestep based on the relative timing of pre- and
//! post-synaptic spikes carried by exponentially-decaying eligibility traces.
//!
//! * [`stdp`] — pair-based STDP (Bi & Poo 1998).
//! * [`triplet_stdp`] — triplet-rule STDP with longer post-synaptic traces
//!   (Pfister & Gerstner 2006), captures higher-order frequency dependence.
//! * [`r_stdp`] — reward-modulated STDP (Florian 2007, Izhikevich 2007) using
//!   slow eligibility traces gated by a global reward signal.

/// BCM sliding-threshold rule and Oja Hebbian PCA rule (homeostatic plasticity).
pub mod homeostatic;
/// Reward-modulated STDP using eligibility traces.
pub mod r_stdp;
/// Pair-based spike-timing dependent plasticity.
pub mod stdp;
/// Triplet STDP with longer post-synaptic traces (Pfister-Gerstner 2006).
pub mod triplet_stdp;

pub use homeostatic::{
    BcmConfig, BcmState, OjaConfig, bcm_equilibrium_theta, bcm_run, bcm_step, oja_batch,
    oja_explained_variance, oja_normalize, oja_step,
};

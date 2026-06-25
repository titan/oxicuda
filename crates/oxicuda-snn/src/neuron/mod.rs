//! Spiking neuron models.
//!
//! Each module exposes a `*Config`, `*State`, and a step function that updates
//! the state in place and writes binary spikes to an output buffer. All numerics
//! are `f32`; integration uses explicit Euler unless otherwise noted.

/// Adaptive Exponential Integrate-and-Fire neuron (Brette-Gerstner 2005).
pub mod adex;
/// Adaptive-threshold Leaky Integrate-and-Fire neuron (Bellec et al. 2018).
pub mod alif;
/// Two-layer nonlinear dendritic neuron (Poirazi-Brannon-Mel 2003).
pub mod dendritic;
/// Event-driven LIF simulation backend for very sparse spiking regimes.
pub mod event_driven;
/// Heterogeneous LIF population with per-neuron `τ_m` and `v_th`.
pub mod het_lif;
/// Hodgkin-Huxley and Pinsky-Rinzel conductance-based neuron models.
pub mod hodgkin_huxley;
/// Pure Integrate-and-Fire neuron (no leak).
pub mod integrate_fire;
/// Izhikevich neuron (2003) with quadratic + recovery dynamics.
pub mod izhikevich;
/// Leaky Integrate-and-Fire neuron.
pub mod lif;
/// Stochastic Poisson rate neuron.
pub mod poisson;
/// Quadratic Integrate-and-Fire and canonical Theta neuron (Ermentrout-Kopell).
pub mod qif;
/// Leaky Integrate-and-Fire neuron with an absolute refractory period.
pub mod refractory;
/// Spike Response Model (SRM₀) kernel-based neuron (Gerstner-Kistler 2002).
pub mod srm;

pub use dendritic::{DendriticNeuron, DendriticSubunit, sigmoid as dendritic_sigmoid};
pub use event_driven::{EventDrivenLif, SpikeRecord, SynapticEvent, clock_stepped_spike_times};
pub use hodgkin_huxley::{HhConfig, HhState, PrConfig, PrState, hh_run, hh_step, pr_step};
pub use qif::{
    QifConfig, QifState, ThetaConfig, ThetaState, qif_step, theta_step, theta_to_voltage,
    voltage_to_theta,
};
pub use refractory::{
    RefractoryLifConfig, RefractoryLifState, refractory_lif_step, refractory_lif_step_batch,
};
pub use srm::{
    SrmConfig, SrmState, psp_kernel as srm_psp_kernel, psp_train as srm_psp_train,
    refractory_kernel as srm_refractory_kernel, srm_step,
};

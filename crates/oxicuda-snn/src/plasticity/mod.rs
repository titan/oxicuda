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
//! * [`tempotron`] — supervised binary temporal classifier learning on the peak
//!   sub-threshold voltage (Gütig & Sompolinsky 2006).
//! * [`resume`] — Remote Supervised Method matching an output spike train to a
//!   teacher train via STDP/anti-STDP windows (Ponulak & Kasiński 2010).

/// BCM sliding-threshold rule and Oja Hebbian PCA rule (homeostatic plasticity).
pub mod homeostatic;
/// Intrinsic plasticity adapting transfer-function gain and bias (Triesch 2005).
pub mod intrinsic;
/// Reward-modulated STDP using eligibility traces.
pub mod r_stdp;
/// ReSuMe supervised spike-train learning (Ponulak-Kasiński 2010).
pub mod resume;
/// Pair-based spike-timing dependent plasticity.
pub mod stdp;
/// Tempotron binary temporal spike classifier (Gütig-Sompolinsky 2006).
pub mod tempotron;
/// Triplet STDP with longer post-synaptic traces (Pfister-Gerstner 2006).
pub mod triplet_stdp;

pub use homeostatic::{
    BcmConfig, BcmState, OjaConfig, bcm_equilibrium_theta, bcm_run, bcm_step, oja_batch,
    oja_explained_variance, oja_normalize, oja_step,
};
pub use intrinsic::{IpConfig, IpState, ip_activation, ip_run, ip_step};
pub use resume::{
    ResumeConfig, ResumeState, learning_window as resume_learning_window, resume_decay,
    resume_step, resume_step_multi,
};
pub use tempotron::{
    Tempotron, TempotronConfig, kernel_norm as tempotron_kernel_norm,
    kernel_peak_time as tempotron_kernel_peak_time, psp_kernel as tempotron_psp_kernel,
};

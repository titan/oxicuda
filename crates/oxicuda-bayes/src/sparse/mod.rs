//! Sparse Bayesian models with sparsity-inducing priors.
//!
//! * **horseshoe** — the global-local shrinkage prior (Carvalho, Polson &
//!   Scott, 2009/2010), a continuous scale-mixture with a half-Cauchy =
//!   inverse-gamma augmentation and a full Gibbs sampler.
//! * **spike_slab** — the discrete point-mass spike-and-slab prior (George &
//!   McCulloch 1997) with a collapsed Gibbs sampler that returns posterior
//!   inclusion probabilities for Bayesian variable selection.

pub mod horseshoe;
pub mod spike_slab;

pub use horseshoe::{
    HorseshoeConfig, HorseshoeFit, HorseshoeRegression, half_cauchy_log_pdf, horseshoe_log_pdf,
    ridge_regression, shrinkage_factor,
};
pub use spike_slab::{SpikeSlabConfig, SpikeSlabFit, SpikeSlabRegression};

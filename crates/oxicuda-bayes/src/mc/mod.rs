//! Monte Carlo methods: Sequential Monte Carlo / Bootstrap Particle Filter,
//! and MCMC convergence diagnostics.
pub mod convergence_diagnostics;
pub mod smc;

pub use convergence_diagnostics::{
    ConvergenceSummary, GewekeConfig, diagnose, effective_sample_size as ess_from_chain, geweke_z,
    multi_chain_ess, r_hat,
};
pub use smc::{
    LcgRng, SmcConfig, SmcState, effective_sample_size, smc_filter, smc_init, smc_mean, smc_step,
    smc_variance, systematic_resample,
};

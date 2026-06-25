//! Monte Carlo methods: Sequential Monte Carlo / Bootstrap Particle Filter,
//! MCMC convergence diagnostics, conjugate Bayesian updates, and predictive
//! model-selection criteria (WAIC / PSIS-LOO / DIC).
pub mod conjugate;
pub mod convergence_diagnostics;
pub mod model_selection;
pub mod smc;

pub use conjugate::{
    BetaPosterior, DirichletPosterior, GammaPosterior, NormalInverseGamma, NormalKnownVarPosterior,
};
pub use convergence_diagnostics::{
    ConvergenceSummary, GewekeConfig, diagnose, effective_sample_size as ess_from_chain, geweke_z,
    multi_chain_ess, r_hat,
};
pub use model_selection::{
    DicResult, PointwiseLpd, PsisLooResult, WaicResult, compare_elpd, dic, pointwise_lpd, psis_loo,
    waic,
};
pub use smc::{
    LcgRng, SmcConfig, SmcState, effective_sample_size, smc_filter, smc_init, smc_mean, smc_step,
    smc_variance, systematic_resample,
};

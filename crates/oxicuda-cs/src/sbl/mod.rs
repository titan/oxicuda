//! Sparse Bayesian Learning algorithms.

pub mod fast_marginal_likelihood;
pub mod relevance_vector;
pub mod smc_cs;
pub mod sparse_bayesian;

pub use fast_marginal_likelihood::fast_marginal_likelihood;
pub use relevance_vector::{Rvm, RvmConfig, RvmFit, RvmKernel, rvm_fit_design};
pub use smc_cs::{SmcCs, SmcCsConfig};
pub use sparse_bayesian::sparse_bayesian;

/// SBL result.
#[derive(Debug, Clone)]
pub struct SblResult {
    pub x: Vec<f64>,
    /// Estimated hyperparameter inverse variances `γ_j = 1/α_j`.
    pub gamma: Vec<f64>,
    pub sigma2: f64,
    pub iterations: usize,
}

//! Bayesian survival analysis via Markov Chain Monte Carlo (MCMC).
//!
//! Supports three model families:
//! - **Weibull** parametric model with Metropolis-Hastings in log-space.
//! - **Log-normal** parametric model with Metropolis-Hastings.
//! - **Cox-Bayes** semi-parametric Cox model with normal prior on β.
//!
//! All models use random-walk Metropolis-Hastings with optional adaptive step size
//! targeting acceptance rate ~0.234 (optimal for d-dimensional Gaussian targets).

pub mod mcmc_survival;

pub use mcmc_survival::{
    BayesSurvModel, CoxBayes, LogNormalBayes, McmcChain, McmcConfig, WeibullBayes, compute_dic,
    cox_bayes, log_normal_bayes, posterior_predictive_survival, weibull_bayes,
};

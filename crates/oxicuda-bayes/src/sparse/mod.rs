//! Sparse Bayesian models with global-local shrinkage priors.
//!
//! Currently provides the **horseshoe** prior (Carvalho, Polson & Scott,
//! 2009/2010) for sparse linear regression, including its half-Cauchy =
//! inverse-gamma scale-mixture augmentation and a full Gibbs sampler.

pub mod horseshoe;

pub use horseshoe::{
    HorseshoeConfig, HorseshoeFit, HorseshoeRegression, half_cauchy_log_pdf, horseshoe_log_pdf,
    ridge_regression, shrinkage_factor,
};

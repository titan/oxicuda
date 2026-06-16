//! Copula models for multivariate survival analysis.
//!
//! Currently provides bivariate copula fitting using the IFM estimator
//! with Frank, Clayton, and Gumbel families.

pub mod bivariate;

pub use bivariate::{
    BivariateCopulaFit, CopulaConfig, CopulaFamily, WeibullMarginalFit, copula_survival_prob,
    fit_bivariate_copula, kendall_tau_from_theta, theta_from_kendall_tau,
};

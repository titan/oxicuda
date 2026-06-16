//! Zero-cost proxies for predictor-free architecture ranking.
//!
//! - [`zero_cost`] — NASWOT logdet kernel + SNIP / GraSP / SynFlow saliencies,
//!   plus the [`ZeroCostProxy`] selector and [`rank_architectures`] helper.
//! - [`jacobian_covariance`] — the original *jacov* score: the negative
//!   sum of `ln(λ + ε) + 1/(λ + ε)` over the eigenvalues of the Jacobian-row
//!   correlation matrix (Mellor 2020), with a self-contained symmetric Jacobi
//!   eigensolver.
//!
//! Unlike the surrogate predictors in [`crate::predictor`], these proxies need
//! no training data and no fitted model: they score an architecture directly
//! from forward-only (and, for SNIP / GraSP / SynFlow / jacov, single-backward)
//! signals gathered on an untrained network.

pub mod jacobian_covariance;
pub mod zero_cost;

pub use jacobian_covariance::{
    JACOV_EPSILON, jacobian_covariance_score, pearson_correlation_matrix, symmetric_eigenvalues,
};
pub use zero_cost::{
    NASWOT_RIDGE, ZeroCostProxy, grasp_score, naswot_score, rank_architectures, snip_score,
    synflow_score,
};

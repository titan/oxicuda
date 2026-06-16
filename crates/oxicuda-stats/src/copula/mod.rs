//! Copula models for dependence-structure modelling.
//!
//! - [`copulas`] — bivariate Gaussian/Clayton/Frank/Gumbel families fitted by
//!   method-of-moments via Kendall's τ.
//! - [`gaussian_copula`] — the *multivariate* Gaussian copula with a full
//!   correlation matrix Σ, empirical-CDF pseudo-observations and ML/MoM fitting.
//! - [`archimedean`] — Archimedean copulas characterised by their generator
//!   φ(t), with maximum-likelihood parameter estimation.
//! - [`vine`] — pair-copula constructions (C-vines and D-vines): vine density
//!   evaluation and sequential tree-by-tree estimation built from the bivariate
//!   pair-copulas above.

pub mod archimedean;
pub mod copulas;
pub mod gaussian_copula;
pub mod vine;

pub use archimedean::{ArchimedeanCopula, ArchimedeanFamily};
pub use copulas::{
    CopulaFamily, CopulaFit, copula_cdf, copula_fit, copula_log_likelihood, copula_pdf,
    copula_sample, copula_tail_dependence, kendall_tau_pairs,
};
pub use gaussian_copula::{GaussianCopula, pseudo_observations};
pub use vine::{PairCopula, VineCopula, VineFitConfig, VineType, vine_fit};

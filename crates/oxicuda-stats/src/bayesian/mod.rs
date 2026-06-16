//! Bayesian inference primitives.
//!
//! Provides conjugate prior updates, credible intervals (HDI and equal-tails),
//! and Bayes factors via the Savage-Dickey density ratio.
//!
//! # Supported conjugate models
//!
//! | Model                  | Prior             | Likelihood    | Posterior          |
//! |------------------------|-------------------|---------------|--------------------|
//! | Normal-Normal          | N(μ₀, τ₀²)        | N(μ, σ²) known σ | N(μₙ, τₙ²)    |
//! | Normal-InverseGamma    | NIG(m₀, κ₀, α₀, β₀)| N(μ, σ²)   | NIG(mₙ, κₙ, αₙ, βₙ)|
//! | Beta-Binomial          | Beta(α, β)        | Bin(n, p)     | Beta(α+k, β+n-k)   |
//! | Gamma-Poisson          | Gamma(α, β)       | Poisson(λ)    | Gamma(α+Σx, β+n)   |
//! | Dirichlet-Multinomial  | Dirichlet(α)      | Multinomial   | Dirichlet(α+n)     |

pub mod conjugate;
pub mod dirichlet_mult;
pub use conjugate::*;
pub use dirichlet_mult::{DirMultFitConfig, DirichletMultinomial, dirichlet_multinomial_mle};

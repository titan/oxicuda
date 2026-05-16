//! Probability distributions: pdf, cdf, ppf (quantile/inverse cdf).
//!
//! All implementations are pure Rust and return `StatsResult` for fallible operations.

pub mod beta;
pub mod binomial;
pub mod chi_squared;
pub mod exponential;
pub mod f_dist;
pub mod gamma;
pub mod normal;
pub mod poisson;
pub mod student_t;

pub use beta::Beta;
pub use binomial::Binomial;
pub use chi_squared::ChiSquared;
pub use exponential::Exponential;
pub use f_dist::FDist;
pub use gamma::Gamma;
pub use normal::Normal;
pub use poisson::Poisson;
pub use student_t::StudentT;

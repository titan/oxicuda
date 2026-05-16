//! Special mathematical functions used throughout statistical computations.
//!
//! All implementations are pure Rust with no external dependencies.

pub mod betainc;
pub mod digamma;
pub mod erf;
pub mod gammaln;
pub mod lgamma_series;

pub use betainc::{betainc, gammp, gammq};
pub use digamma::digamma;
pub use erf::{erf, erfc, erfinv};
pub use gammaln::{beta_log, lgamma};
pub use lgamma_series::{lgamma_series, stirling_series};

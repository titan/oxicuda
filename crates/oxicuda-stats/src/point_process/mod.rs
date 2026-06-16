//! Temporal point processes.
//!
//! Currently provides the univariate exponential-kernel **Hawkes** self-exciting
//! process: conditional-intensity evaluation, the exact O(n) recursive
//! log-likelihood (Ogata, 1981), maximum-likelihood estimation, and Ogata
//! thinning simulation.

pub mod hawkes;

pub use hawkes::{
    HawkesMleConfig, HawkesMleResult, HawkesParams, hawkes_compensator, hawkes_intensity,
    hawkes_log_likelihood, hawkes_log_likelihood_naive, hawkes_mle, hawkes_simulate,
};

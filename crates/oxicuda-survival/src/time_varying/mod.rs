//! Time-varying covariates / counting-process formulation of Cox regression.

pub mod counting_process;
pub mod time_varying_cox;

pub use counting_process::{CountingInterval, CountingProcessDataset};
pub use time_varying_cox::{TvCoxFit, fit_time_varying_cox};

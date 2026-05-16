//! Correlation coefficients with inference.

pub mod kendall_tau;
pub mod pearson;
pub mod spearman;

pub use kendall_tau::{KendallResult, kendall_tau};
pub use pearson::{PearsonResult, pearson_r};
pub use spearman::{SpearmanResult, spearman_rho};

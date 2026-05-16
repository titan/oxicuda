//! Confidence intervals: normal, t, bootstrap, proportion.

pub mod bootstrap_ci;
pub mod normal_ci;
pub mod proportion_ci;
pub mod t_ci;

pub use bootstrap_ci::{bca_ci, percentile_ci};
pub use normal_ci::normal_ci;
pub use proportion_ci::{agresti_coull_ci, clopper_pearson_ci, wilson_ci};
pub use t_ci::t_ci;

/// Confidence interval result.
#[derive(Debug, Clone, Copy)]
pub struct CiResult {
    pub lower: f64,
    pub upper: f64,
}

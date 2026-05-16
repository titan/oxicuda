//! Restricted mean survival time (RMST).

pub mod restricted_mean;
pub mod rmst_estimator;

pub use restricted_mean::restricted_mean_from_curve;
pub use rmst_estimator::{RmstResult, rmst_from_dataset};

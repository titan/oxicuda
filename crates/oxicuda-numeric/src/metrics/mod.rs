//! Numerical diagnostics: error norms, condition number, residuals.

pub mod metrics;

pub use metrics::{
    absolute_error, condition_number_two_by_two, max_norm, relative_error, residual_norm,
};

//! Summary metrics: median survival, restricted mean, S(τ) at horizons.

pub mod metrics;

pub use metrics::{median_survival, restricted_mean_metric, survival_at_horizon};

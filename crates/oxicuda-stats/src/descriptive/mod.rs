//! Descriptive statistics: location, dispersion, robust measures, and quantiles.

pub mod quantile;
pub mod robust;
pub mod summary;

pub use quantile::{percentile, quantile, quantile_inclusive};
pub use robust::{iqr, mad, trimmed_mean, winsorized_mean};
pub use summary::{kurtosis, mean, sample_std, sample_var, skewness, std_dev, variance};

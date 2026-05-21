//! Descriptive statistics: location, dispersion, robust measures, and quantiles.

pub mod lmoments;
pub mod quantile;
pub mod robust;
pub mod summary;

pub use lmoments::{
    LMoments, l_moment_1, l_moment_2, l_moment_3, l_moment_4, l_moments, pwm, trimmed_mean_lm,
    trimmed_std_error, trimmed_variance, winsorised_mean_lm,
};
pub use quantile::{percentile, quantile, quantile_inclusive};
pub use robust::{iqr, mad, trimmed_mean, winsorized_mean};
pub use summary::{kurtosis, mean, sample_std, sample_var, skewness, std_dev, variance};

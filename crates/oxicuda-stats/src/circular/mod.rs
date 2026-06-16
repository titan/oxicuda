//! Circular statistics for directional data (angles in radians).
//!
//! Provides the Von Mises distribution (PDF, CDF, MLE) and the Rayleigh
//! test of circular uniformity, together with standard circular summary
//! statistics (mean direction, variance, standard deviation).

pub mod circular;
pub mod circular_stats;

pub use circular::{
    CircularError, CircularResult, RayleighResult, VonMisesFit, circular_mean, circular_std,
    circular_variance, rayleigh_test, von_mises_cdf, von_mises_mle, von_mises_pdf,
};
pub use circular_stats::{
    MeanDirection, WatsonWilliamsResult, kappa_mle, kappa_mle_from_angles, mean_direction,
    watson_williams_test,
};

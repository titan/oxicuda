//! Extreme-value statistics.
//!
//! Provides the [`extreme_value`] module with the Generalised Extreme Value
//! (GEV) distribution for block maxima and the Generalised Pareto Distribution
//! (GPD) for peaks-over-threshold exceedances, including CDF/PDF/quantile
//! evaluation, Probability-Weighted-Moment fitting, and return-level computation.
//!
//! # References
//! - Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme
//!   Values*, Springer.
//! - Hosking, J. R. M. & Wallis, J. R. (1987). "Parameter and Quantile Estimation
//!   for the Generalized Pareto Distribution." *Technometrics* 29(3):339-349.

pub mod extreme_value;

pub use extreme_value::{Gev, Gpd};

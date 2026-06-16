//! Classical exponential-smoothing forecasters.
//!
//! - [`holt_winters`] — Holt-Winters triple exponential smoothing (additive /
//!   multiplicative seasonality, optional damped trend).
//! - [`croston`] — Croston's method and its SBA / TSB variants for intermittent
//!   demand forecasting.

pub mod croston;
pub mod holt_winters;

pub use croston::{Croston, CrostonConfig, CrostonMethod};
pub use holt_winters::{HoltWinters, HoltWintersConfig, Seasonality};

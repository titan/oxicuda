//! Multi-task time-series heads.
//!
//! * [`forecast_classify`] — joint horizon-forecast + series-classification from
//!   a shared encoder with a `λ`-weighted MSE + cross-entropy objective.

pub mod forecast_classify;

pub use forecast_classify::{BackboneWeights, HeadWeights, MultiTaskConfig, MultiTaskForecaster};

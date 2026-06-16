//! Online / streaming forecasting utilities.
//!
//! These forecasters consume observations one at a time and update their
//! parameters recursively, without re-fitting from scratch — suitable for
//! low-latency streaming inference.

pub mod streaming_forecast;

pub use streaming_forecast::StreamingForecaster;

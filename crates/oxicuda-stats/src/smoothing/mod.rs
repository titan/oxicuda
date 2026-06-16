//! Exponential smoothing models for time series forecasting.
//!
//! This module provides the Holt-Winters family of ETS (Error-Trend-Seasonal) models:
//! - **Simple** (SES): level only
//! - **Double**: level + linear trend (Holt)
//! - **Additive**: level + trend + additive seasonality (Winters)
//! - **Multiplicative**: level + trend + multiplicative seasonality (Winters)

pub mod holt_winters;

pub use holt_winters::{
    HwConfig, HwResult, HwVariant, hw_aic, hw_bic, hw_fit, hw_forecast, hw_residuals,
};

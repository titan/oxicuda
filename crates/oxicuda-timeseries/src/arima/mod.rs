//! ARIMA-family models for classical time-series forecasting.
//!
//! Provides Box-Jenkins SARIMA(p,d,q)×(P,D,Q)_s for seasonal modelling.

pub mod sarima;
pub use sarima::{Sarima, SarimaConfig};

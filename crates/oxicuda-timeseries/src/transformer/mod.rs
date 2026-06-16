//! Transformer-based time-series architectures for OxiCUDA.
//!
//! Modules in this directory implement attention mechanisms and gated networks
//! specific to time-series forecasting and anomaly detection.

pub mod anomaly;
pub mod autocorrelation;
pub mod tft_vsn;

pub use anomaly::{AnomalyConfig, AnomalyResult, AnomalyTransformer};
pub use autocorrelation::{AutocorrConfig, AutocorrelationBlock};
pub use tft_vsn::{Grn, VariableSelectionNet, VsnConfig};

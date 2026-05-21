//! Time-series anomaly detection.
//!
//! # Algorithms
//!
//! - [`lstm_ae`] — RNN Autoencoder (Malhotra et al. 2016): encodes a fixed-length window
//!   to a context vector, then decodes in reverse order; anomaly score = mean per-timestep
//!   reconstruction MSE over all windows containing that timestep.
pub mod lstm_ae;
pub use lstm_ae::*;

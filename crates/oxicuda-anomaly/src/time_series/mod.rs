//! Time-series anomaly detection.
//!
//! # Algorithms
//!
//! - [`lstm_ae`] — RNN Autoencoder (Malhotra et al. 2016): encodes a fixed-length window
//!   to a context vector, then decodes in reverse order; anomaly score = mean per-timestep
//!   reconstruction MSE over all windows containing that timestep.
//! - [`mod@spectral_residual`] — Spectral Residual (Ren et al. 2019): saliency-based detection
//!   via the residual of the log-amplitude Fourier spectrum.
pub mod lstm_ae;
pub mod spectral_residual;

pub use lstm_ae::*;
pub use spectral_residual::{SpectralResidualConfig, SpectralResidualResult, spectral_residual};

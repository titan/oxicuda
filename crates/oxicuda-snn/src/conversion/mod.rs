//! ANN→SNN conversion utilities.
//!
//! Provides rate-coded conversion of pre-trained ReLU artificial neural networks
//! into integrate-and-fire spiking equivalents via per-layer threshold balancing.
//! The 99th-percentile activation method (Rueckauer et al., 2017) is implemented
//! end-to-end so that arbitrary feed-forward chains can be rescaled in one pass.

/// ANN→SNN rate-coded conversion via threshold balancing.
pub mod ann2snn;
/// Layer-wise threshold balancing (99-percentile activation method).
pub mod threshold_balance;

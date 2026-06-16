//! Input-feature embeddings for coordinate networks and PINNs.
//!
//! These are standalone, network-agnostic coordinate lifts that mitigate the
//! spectral bias of multi-layer perceptrons on multi-scale problems.

pub mod fourier_features;

pub use fourier_features::{FourierFeatureEmbeddingConfig, FourierFeatures};

//! Graph-based anomaly detection.
//!
//! # Algorithms
//!
//! - [`dominant`] — DOMINANT: Deep Autoencoder-based anomaly detection on attributed graphs
//!   (Ding et al. 2019, SDM). Jointly reconstructs adjacency structure and node features.
pub mod dominant;
pub use dominant::*;

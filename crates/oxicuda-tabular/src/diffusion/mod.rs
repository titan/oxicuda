//! Diffusion models for tabular data.
//!
//! Currently provides TabDDPM: a Gaussian denoising diffusion probabilistic
//! model adapted for tabular data (Kotelnikov et al., 2023).

pub mod tabddpm;

pub use tabddpm::{DenoisingMlp, TabDdpm, TabDdpmConfig};

//! Variational autoencoders for tabular data.
//!
//! Currently provides TVAE: the tabular VAE baseline from Xu et al. (2019),
//! with mode-aware Gaussian / categorical decoding and a closed-form KL term.

pub mod tvae;

pub use tvae::{Tvae, TvaeConfig, kl_divergence_standard};

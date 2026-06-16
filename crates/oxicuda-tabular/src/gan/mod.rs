//! Generative adversarial networks for tabular data.
//!
//! Currently provides CTGAN: the conditional tabular GAN of Xu et al. (2019)
//! with mode-specific normalisation, log-frequency conditional sampling, a
//! residual generator with Gumbel-softmax outputs, and a PacGAN discriminator.

pub mod ctgan;

pub use ctgan::{ColumnModes, ConditionalSampler, CtGan, CtganConfig, ModeNormalizer};

//! VAE (Variational Autoencoder) module.
//!
//! Provides Gaussian latent space, encoder, decoder, and VQ-VAE
//! vector quantisation components.

pub mod decoder;
pub mod encoder;
pub mod kl;
pub mod quantize;

pub use decoder::{Decoder, DecoderConfig, DecoderWeights};
pub use encoder::{Encoder, EncoderConfig, EncoderWeights, ResidualBlock};
pub use kl::GaussianLatent;
pub use quantize::VqCodebook;

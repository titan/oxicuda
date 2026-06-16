//! VAE (Variational Autoencoder) module.
//!
//! Provides Gaussian latent space, encoder, decoder, VQ-VAE
//! vector quantisation components, MLP β-VAE encoder, and
//! hierarchical VQ-VAE-2.

pub mod decoder;
pub mod ema_codebook;
pub mod encoder;
pub mod kl;
pub mod latent_encoder;
pub mod quantize;
pub mod vq_vae2;

pub use decoder::{Decoder, DecoderConfig, DecoderWeights};
pub use ema_codebook::{EmaCodebook, EmaCodebookConfig};
pub use encoder::{Encoder, EncoderConfig, EncoderWeights, ResidualBlock};
pub use kl::GaussianLatent;
pub use latent_encoder::{VaeEncoder, VaeEncoderConfig};
pub use quantize::VqCodebook;
pub use vq_vae2::{VqCodebookEma, VqVae2, VqVae2Config};

//! Generative neural fields.
//!
//! - `pi_gan`: pi-GAN periodic-implicit (FiLM-SIREN) generative radiance field
//!   conditioned on a latent code.

pub mod pi_gan;

pub use pi_gan::{FilmParams, PiGan, PiGanConfig};

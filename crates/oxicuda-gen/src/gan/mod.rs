//! GAN (Generative Adversarial Network) building blocks.
//!
//! Currently provides the alias-free operators and generator stubs from
//! StyleGAN3 (Karras et al., 2021).

pub mod stylegan3;

pub use stylegan3::{
    AliasFreeOps, MappingNetwork, StyleGan3Config, StyleGan3Generator, SynthesisLayer,
    kaiser_lowpass_fir, leaky_relu,
};

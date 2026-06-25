//! Cepstral analysis — real / power / complex cepstrum, liftering, and
//! cepstral pitch estimation.
//!
//! The cepstrum is the (inverse) Fourier transform of the log spectrum; it
//! turns a convolution of source and filter into an addition, enabling
//! homomorphic deconvolution, pitch detection and echo location.

pub mod cepstrum;

pub use cepstrum::{
    cepstral_pitch, complex_cepstrum, highpass_lifter, inverse_complex_cepstrum, lowpass_lifter,
    power_cepstrum, real_cepstrum, sinusoidal_lifter, unwrap_phase,
};

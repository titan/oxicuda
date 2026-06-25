//! Text-to-speech acoustic-model synthesis for `oxicuda-audio`.
//!
//! Provides:
//! - **`fastspeech2`**: The deterministic FastSpeech 2 acoustic-model core
//!   (Ren et al. 2021) — a feed-forward Transformer (FFT) encoder/decoder with
//!   the variance adaptor (duration predictor + length regulator + pitch /
//!   energy predictors) mapping phoneme embeddings to a mel spectrogram.
//! - **`vits2`**: The VITS / VITS2 conditional-VAE-with-normalising-flow
//!   acoustic model (Kim 2021 / Kong 2023) — an invertible prior flow
//!   (`Vits2Flow`), a flow-based stochastic duration predictor
//!   (`StochasticDurationPredictor`), posterior + prior encoders with the VITS
//!   ELBO KL terms and monotonic alignment search, wired into a `Vits2` model
//!   with analysis (teacher) and inference (synthesis) passes. Neural waveform
//!   synthesis lives in [`crate::vocoder`].

pub mod fastspeech2;
pub mod vits2;

pub use fastspeech2::{
    ConvFfnWeights, DurationPredictor, FastSpeech2, FastSpeech2Config, FftBlock, SelfAttnWeights,
    VariancePredictor, embed_and_add, length_regulate, length_regulate_with_pace, quantize_to_bins,
};
pub use vits2::{
    ActNorm, AffineCoupling, PosteriorEncoder, PriorEncoder, RationalQuadraticSpline,
    RqSplineCoupling, StochasticDurationPredictor, Vits2, Vits2Analysis, Vits2Config, Vits2Flow,
    flow_kl, gaussian_kl, monotonic_alignment_search, reparameterize,
};

//! Time-scale modification: change a signal's duration without altering pitch
//! (and pitch without altering duration).
//!
//! - [`phase_vocoder`]: STFT phase vocoder (Flanagan-Golden 1966,
//!   Laroche-Dolson 1999) — time-stretch via per-bin instantaneous-frequency
//!   phase propagation, plus pitch-shift via stretch + resample.

pub mod phase_vocoder;

pub use phase_vocoder::{
    PhaseVocoderConfig, instantaneous_frequency, phase_vocoder_stretch, pitch_shift,
    resample_linear,
};

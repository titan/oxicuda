//! Spectral estimation: power spectral density (PSD) estimators.
//!
//! This module provides classic non-parametric PSD estimators — the
//! periodogram, Welch's averaged periodogram, Bartlett's method, and the
//! sine-taper multitaper estimator — all returning one-sided densities that
//! integrate to signal power (Parseval-calibrated).

pub mod goertzel;
pub mod welch;

pub use goertzel::{
    DTMF_FREQUENCIES_HZ, GoertzelBin, dtmf_decode, goertzel, goertzel_hz, goertzel_power_spectrum,
};
pub use welch::{PsdScaling, bartlett_psd, multitaper_psd, periodogram, welch};
